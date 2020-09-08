#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-25
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from typing import Optional
import torch
import numpy as np


def dh_transformation(theta: torch.Tensor, alpha: torch.Tensor, a: torch.Tensor, d: torch.Tensor,
                      theta_offset: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Given DH parameters and batched joint values, compute batched transformations matrices 4x4.
    :param theta [BxNDOF]
    :param alpha [NDOF]
    :param a [NDOF]
    :param d [NDOF]
    :param theta_offset [NDOF]
    :returns [B x NDOF x 4 x 4]
    """
    assert len(theta.shape) == 2
    assert len(alpha.shape) == 1
    b = theta.shape[0]
    n = theta.shape[1]

    tz = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, n, 1, 1)  # [B x N x 4 x 4]
    tz[:, :, 2, 3] = d

    tx = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, n, 1, 1)  # [B x N x 4 x 4]
    tx[:, :, 0, 3] = a

    th = theta if theta_offset is None else theta + theta_offset
    thc = th.cos()
    ths = th.sin()
    rz = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, n, 1, 1)  # [B x N x 4 x 4]
    rz[:, :, 0, 0] = thc
    rz[:, :, 0, 1] = -ths
    rz[:, :, 1, 0] = ths
    rz[:, :, 1, 1] = thc

    rx = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, n, 1, 1)  # [B x N x 4 x 4]
    alc = alpha.cos()
    als = alpha.sin()
    rx[:, :, 1, 1] = alc
    rx[:, :, 1, 2] = -als
    rx[:, :, 2, 1] = als
    rx[:, :, 2, 2] = alc

    return tz.matmul(rz).matmul(rx).matmul(tx)


def forward_kinematic(transformations: torch.Tensor):
    """ From transformations [B x N x 4 x 4] computes forward kinematic [B x 4 x 4]"""
    fk = transformations[:, 0, :, :].clone()
    for k in range(1, transformations.shape[1]):
        fk = fk.matmul(transformations[:, k, :, :])
    return fk


def cumulative_transformations(rel_transformations: torch.Tensor):
    """ Return transformation of each link w.r.t. world. """
    cum_transformations = rel_transformations[:, :, :, :].clone()
    for k in range(1, rel_transformations.shape[1]):
        cum_transformations[:, k, :, :] = cum_transformations[:, k - 1, :, :].bmm(rel_transformations[:, k, :, :])
    return cum_transformations


def jacobian(cum_transformations: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian matrix J, that solves equation dx/dt = J dq/dt.
    """
    b = cum_transformations.shape[0]
    n = cum_transformations.shape[1]
    jac = torch.zeros(b, 6, n)
    for k in range(0, n):
        tran = cum_transformations[:, k - 1] if k > 0 else torch.eye(4).unsqueeze(0).repeat(b, 1, 1)
        n = tran[:, :3, :3].matmul(torch.tensor([0., 0., 1.]).reshape(3, 1)).squeeze(-1)  # [B x 3]
        r_tmp = cum_transformations[:, -1, :3, 3] - tran[:, :3, 3]  # [B x 3]
        dr = n.cross(r_tmp, dim=-1)  # [B x 3]
        jac[:, :3, k] = dr
        jac[:, 3:, k] = n
    return jac


def panda_dh_parameters():
    """ Get DH parameters for panda robot. """
    a = torch.tensor([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
    d = torch.tensor([0.333, 0., 0.316, 0., 0.384, 0., 0.])
    alpha = torch.tensor([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.])
    theta = torch.zeros(7)
    return alpha, a, d, theta

# def ur5_dh_parameters():
#     """ Get DH parameters for panda robot. """
#     a = torch.tensor([0., -0.425, -0.39225, 0., 0., 0.])
#     d = torch.tensor([0.1625, 0., 0., 0.1333, 0.0997, 0.0996])
#     alpha = torch.tensor([np.pi / 2, 0., 0., np.pi / 2, -np.pi / 2, 0.])
#     theta = torch.zeros(6)
#     return alpha, a, d, theta

def ur5_dh_parameters():
    """ Get DH parameters for panda robot. """
    a = torch.tensor([0., -0.425, -0.39225, 0., 0., 0.])
    d = torch.tensor([0.089159, 0., 0., 0.10915, 0.09465, 0.0823])
    alpha = torch.tensor([np.pi / 2, 0., 0., np.pi / 2, -np.pi / 2, 0.])
    theta = torch.zeros(6)
    # theta[0] = np.pi
    return alpha, a, d, theta

