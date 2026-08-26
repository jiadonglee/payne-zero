"""Differentiable atmosphere iteration in PyTorch.

A batched, autograd-capable twin of the ``payne_zero_atmosphere`` iteration,
built on the torch physics already in ``payne_zero_synthesis``. It exists so the
atmosphere initializer can be trained *through* solver iterations rather than
against converged profiles.

This is a twin, not the certified solver. Every claim about iteration counts is
measured by running the real ``payne_zero_atmosphere`` path (see ``bench/``);
numbers produced here are for optimization, not for reporting.
"""
