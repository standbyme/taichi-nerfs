import os
import torch
import numpy as np

import taichi as ti
from taichi.math import uvec3

data_type = ti.f32
torch_type = torch.float32

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3 / 1024
SQRT3_2 = 1.7320508075688772 * 2


def res_in_level_np(
        level_i, 
        base_res, 
        log_per_level_scale
    ):
    result = np.ceil(
        float(base_res) * np.exp(
            float(level_i) * log_per_level_scale
        ) - 1.0
    )
    return float(result + 1)

def scale_in_level_np(
        base_res, 
        max_res,
        levels,
    ):
    result = np.log(
        float(max_res) / float(base_res)
    ) / float(levels - 1)
    return result

def align_to(x, y):
    return int((x+y-1)/y)*y

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

@ti.func
def scalbn(x, exponent):
    return x * ti.math.pow(2, exponent)


@ti.func
def calc_dt(t, exp_step_factor, grid_size, scale):
    return ti.math.clamp(t * exp_step_factor, SQRT3_MAX_SAMPLES,
                         SQRT3_2 * scale / grid_size)


@ti.func
def frexp_bit(x):
    exponent = 0
    if x != 0.0:
        # frac = ti.abs(x)
        bits = ti.bit_cast(x, ti.u32)
        exponent = ti.i32((bits & ti.u32(0x7f800000)) >> 23) - 127
        # exponent = (ti.i32(bits & ti.u32(0x7f800000)) >> 23) - 127
        bits &= ti.u32(0x7fffff)
        bits |= ti.u32(0x3f800000)
        frac = ti.bit_cast(bits, ti.f32)
        if frac < 0.5:
            exponent -= 1
        elif frac > 1.0:
            exponent += 1
    return exponent


@ti.func
def mip_from_pos(xyz, cascades):
    mx = ti.abs(xyz).max()
    # _, exponent = _frexp(mx)
    exponent = frexp_bit(ti.f32(mx)) + 1
    # frac, exponent = ti.frexp(ti.f32(mx))
    return ti.min(cascades - 1, ti.max(0, exponent))


@ti.func
def mip_from_dt(dt, grid_size, cascades):
    # _, exponent = _frexp(dt*grid_size)
    exponent = frexp_bit(ti.f32(dt * grid_size))
    # frac, exponent = ti.frexp(ti.f32(dt*grid_size))
    return ti.min(cascades - 1, ti.max(0, exponent))


@ti.func
def __expand_bits(v):
    v = (v * ti.uint32(0x00010001)) & ti.uint32(0xFF0000FF)
    v = (v * ti.uint32(0x00000101)) & ti.uint32(0x0F00F00F)
    v = (v * ti.uint32(0x00000011)) & ti.uint32(0xC30C30C3)
    v = (v * ti.uint32(0x00000005)) & ti.uint32(0x49249249)
    return v


@ti.func
def __morton3D(xyz):
    xyz = __expand_bits(xyz)
    return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)


@ti.func
def __morton3D_invert(x):
    x = x & (0x49249249)
    x = (x | (x >> 2)) & ti.uint32(0xc30c30c3)
    x = (x | (x >> 4)) & ti.uint32(0x0f00f00f)
    x = (x | (x >> 8)) & ti.uint32(0xff0000ff)
    x = (x | (x >> 16)) & ti.uint32(0x0000ffff)
    return ti.int32(x)


@ti.kernel
def morton3D_invert_kernel(indices: ti.types.ndarray(ndim=1),
                           coords: ti.types.ndarray(ndim=2)):
    for i in indices:
        ind = ti.uint32(indices[i])
        coords[i, 0] = __morton3D_invert(ind >> 0)
        coords[i, 1] = __morton3D_invert(ind >> 1)
        coords[i, 2] = __morton3D_invert(ind >> 2)


def morton3D_invert(indices):
    coords = torch.zeros(indices.size(0),
                         3,
                         device=indices.device,
                         dtype=torch.int32)
    morton3D_invert_kernel(indices.contiguous(), coords)
    ti.sync()
    return coords


@ti.kernel
def morton3D_kernel(xyzs: ti.types.ndarray(ndim=2),
                    indices: ti.types.ndarray(ndim=1)):
    for s in indices:
        xyz = uvec3([xyzs[s, 0], xyzs[s, 1], xyzs[s, 2]])
        indices[s] = ti.cast(__morton3D(xyz), ti.int32)


def morton3D(coords1):
    indices = torch.zeros(coords1.size(0),
                          device=coords1.device,
                          dtype=torch.int32)
    morton3D_kernel(coords1.contiguous(), indices)
    ti.sync()
    return indices


@ti.kernel
def packbits(density_grid: ti.types.ndarray(ndim=1),
             density_threshold: float,
             density_bitfield: ti.types.ndarray(ndim=1)):

    for n in density_bitfield:
        bits = ti.uint8(0)

        for i in ti.static(range(8)):
            bits |= (ti.uint8(1) << i) if (
                density_grid[8 * n + i] > density_threshold) else ti.uint8(0)

        density_bitfield[n] = bits


@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]


@ti.kernel
def torch2ti_vec(field: ti.template(), data: ti.types.ndarray()):
    for I in range(data.shape[0] // 2):
        field[I] = ti.Vector([data[I * 2], data[I * 2 + 1]])


@ti.kernel
def ti2torch_vec(field: ti.template(), data: ti.types.ndarray()):
    for i, j in ti.ndrange(data.shape[0], data.shape[1] // 2):
        data[i, j * 2] = field[i, j][0]
        data[i, j * 2 + 1] = field[i, j][1]


@ti.kernel
def ti2torch_grad_vec(field: ti.template(), grad: ti.types.ndarray()):
    for I in range(grad.shape[0] // 2):
        grad[I * 2] = field.grad[I][0]
        grad[I * 2 + 1] = field.grad[I][1]


@ti.kernel
def torch2ti_grad_vec(field: ti.template(), grad: ti.types.ndarray()):
    for i, j in ti.ndrange(grad.shape[0], grad.shape[1] // 2):
        field.grad[i, j][0] = grad[i, j * 2]
        field.grad[i, j][1] = grad[i, j * 2 + 1]


def save_deployment_model(model, dataset, save_dir):
    padding = torch.zeros(13, 16)
    rgb_out = model.rgb_net.output_layer.weight.detach().cpu()
    rgb_out = torch.cat([rgb_out, padding], dim=0)
    new_dict = {
        'poses': dataset.poses.cpu().numpy(),
        'model.density_bitfield': model.density_bitfield.cpu().numpy(),
        'model.hash_encoder.params': model.pos_encoder.hash_table.detach().cpu().numpy(),
        'model.per_level_scale': model.pos_encoder.log_b,
        'model.xyz_encoder.params': 
            torch.cat(
                [model.xyz_encoder.hidden_layers[0].weight.detach().cpu().reshape(-1),
                model.xyz_encoder.output_layer.weight.detach().cpu().reshape(-1)]
            ).numpy(),
        'model.rgb_net.params': 
            torch.cat(
                [model.rgb_net.hidden_layers[0].weight.detach().cpu().reshape(-1),
                rgb_out.reshape(-1)]
            ).numpy(),
    }
    np.save(
        os.path.join(f'{save_dir}', 'deployment.npy'), 
        new_dict
    )
