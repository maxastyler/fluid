//! This is just a copy of Jos Stam's [paper](https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf)

use ndarray::prelude::*;

#[derive(Copy, Clone, PartialEq)]
pub enum Comp {
    X,
    Y,
    Scalar,
}

/// Set the boundary points
pub fn set_bnd(c: Comp, mut x: ArrayViewMut2<f32>) {
    use Comp::*;
    let n = x.shape().to_vec();
    for i in 1..(n[0] - 1) {
        unsafe {
            *x.uget_mut((i, 0)) = match c {
                X => x[(i, 1)],
                _ => -x[(i, 1)],
            };
            *x.uget_mut((i, n[1] - 1)) = match c {
                X => x[(i, n[1] - 2)],
                _ => -x[(i, n[1] - 2)],
            };
        }
    }
    for i in 1..(n[1] - 1) {
        unsafe {
            *x.uget_mut((0, i)) = match c {
                X => -x[(1, i)],
                _ => x[(1, i)],
            };
            *x.uget_mut((n[0] - 1, i)) = match c {
                X => -x[(n[0] - 2, i)],
                _ => x[(n[0] - 2, i)],
            };
        }
    }
    unsafe {
        *x.uget_mut((0, 0)) = (x[(1, 0)] + x[(0, 1)]) * 0.5;
        *x.uget_mut((n[0] - 1, 0)) = (x[(n[0] - 2, 0)] + x[(n[0] - 1, 1)]) * 0.5;
        *x.uget_mut((n[0] - 1, n[1] - 1)) =
            (x[(n[0] - 2, n[1] - 1)] + x[(n[0] - 1, n[1] - 2)]) * 0.5;
        *x.uget_mut((0, n[1] - 1)) = (x[(1, n[1] - 1)] + x[(0, n[1] - 2)]) * 0.5;
    }
}

/// Diffuse the fluid
pub fn diffuse(
    c: Comp,
    mut x: ArrayViewMut2<f32>,
    x0: ArrayView2<f32>,
    diff: f32,
    dt: f32,
    iterations: usize,
) {
    let n = x.shape().to_vec();
    let a = dt * diff * ((n[0] - 2) * (n[1] - 2)) as f32;
    for _ in 0..iterations {
        for i in 1..(n[0] - 1) {
            for y in 1..(n[1] - 1) {
                unsafe {
                    *x.uget_mut((i, y)) = (x0.uget((i, y))
                        + a * (x.uget((i - 1, y))
                            + x.uget((i + 1, y))
                            + x.uget((i, y - 1))
                            + x.uget((i, y + 1))))
                        / (1.0 + 4.0 * a);
                }
            }
        }
        set_bnd(c, x.view_mut());
    }
}

pub fn bilinear_interp(point: (f32, f32), a: ArrayView2<f32>) -> f32 {
    let x1 = point.0.floor();
    let x2 = point.0.ceil();
    let y1 = point.1.floor();
    let y2 = point.1.ceil();
    let f11 = a[(x1 as usize, y1 as usize)];
    let f12 = a[(x1 as usize, y2 as usize)];
    let f21 = a[(x2 as usize, y1 as usize)];
    let f22 = a[(x2 as usize, y2 as usize)];
    let (x, y) = point;
    let (r1, r2) = if x2 == x1 {
        (f11, f22)
    } else {
        (
            ((x2 - x) / (x2 - x1)) * f11 + ((x - x1) / (x2 - x1)) * f21,
            ((x2 - x) / (x2 - x1)) * f12 + ((x - x1) / (x2 - x1)) * f22,
        )
    };
    if y2 == y1 {
        r1
    } else {
        ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2
    }
}

/// clamp a value between top and bottom
pub fn clamp<T>(x: T, bottom: T, top: T) -> T
where
    T: PartialOrd,
{
    if x > top {
        return top;
    } else if x < bottom {
        bottom
    } else {
        x
    }
}

/// Get the positions that the indices are advected from
pub fn advection_positions(
    mut old_positions: ArrayViewMut3<f32>,
    velocities: ArrayView3<f32>,
    dt: f32,
) {
    let n = old_positions.shape().to_vec();
    let dtx = dt * n[1] as f32;
    let dty = dt * n[2] as f32;
    for i in 1..(n[1] - 1) {
        for j in 1..(n[2] - 1) {
            old_positions[(0, i, j)] = clamp(
                i as f32 - dtx * velocities[(0, i, j)],
                0.5,
                n[1] as f32 - 1.5,
            );
            old_positions[(1, i, j)] = clamp(
                j as f32 - dty * velocities[(1, i, j)],
                0.5,
                n[2] as f32 - 1.5,
            );
        }
    }
}

/// Advect the field using the advection positions
pub fn advect(c: Comp, mut d: ArrayViewMut2<f32>, d0: ArrayView2<f32>, positions: ArrayView3<f32>) {
    let n = d.shape().to_vec();
    for i in 1..(n[0] - 1) {
        for j in 1..(n[1] - 1) {
            d[(i, j)] = bilinear_interp((positions[(0, i, j)], positions[(1, i, j)]), d0);
        }
    }
    set_bnd(c, d);
}

/// project the velocity onto the zero-divergence component, using prev_vel to store density and divergence
pub fn project(mut vel: ArrayViewMut3<f32>, mut prev_vel: ArrayViewMut3<f32>, iterations: usize) {
    // store divergence in prev_vel[0, .., ..]
    let n = vel.shape().to_vec();
    let hx = 1.0 / (n[1] - 2) as f32;
    let hy = 1.0 / (n[2] - 2) as f32;
    for i in 1..(n[1] - 1) {
        for j in 1..(n[2] - 1) {
            prev_vel[(0, i, j)] = -0.5
                * (hx * (vel[(0, i + 1, j)] - vel[(0, i - 1, j)])
                    + hy * (vel[(1, i, j + 1)] - vel[(1, i, j - 1)]));
            prev_vel[(1, i, j)] = 0.0;
        }
    }
    set_bnd(Comp::Scalar, prev_vel.index_axis_mut(Axis(0), 0));
    set_bnd(Comp::Scalar, prev_vel.index_axis_mut(Axis(0), 1));
    for _ in 0..iterations {
        for i in 1..(n[1] - 1) {
            for j in 1..(n[2] - 1) {
                prev_vel[(1, i, j)] = (prev_vel[(0, i, j)]
                    + prev_vel[(1, i + 1, j)]
                    + prev_vel[(1, i - 1, j)]
                    + prev_vel[(1, i, j + 1)]
                    + prev_vel[(1, i, j - 1)])
                    / 4.0;
            }
        }
        set_bnd(Comp::Scalar, prev_vel.index_axis_mut(Axis(0), 1));
    }
    for i in 1..(n[1] - 1) {
        for j in 1..(n[2] - 1) {
            vel[(0, i, j)] -= 0.5 * (prev_vel[(1, i + 1, j)] - prev_vel[(1, i - 1, j)]) / hx;
            vel[(1, i, j)] -= 0.5 * (prev_vel[(1, i, j + 1)] - prev_vel[(1, i, j - 1)]) / hy;
        }
    }
    set_bnd(Comp::X, vel.index_axis_mut(Axis(0), 0));
    set_bnd(Comp::Y, vel.index_axis_mut(Axis(0), 1));
}

/// Assuming the force is stored in `forces`, step the velocity
pub fn vel_step<'a>(
    mut positions: ArrayViewMut3<f32>,
    mut vel: ArrayViewMut3<'a, f32>,
    mut forces: ArrayViewMut3<'a, f32>,
    visc: f32,
    dt: f32,
) {
    vel += &(&forces * dt);
    std::mem::swap(&mut vel, &mut forces);

    diffuse(
        Comp::X,
        vel.index_axis_mut(Axis(0), 0),
        forces.index_axis(Axis(0), 0),
        visc,
        dt,
        20,
    );

    diffuse(
        Comp::Y,
        vel.index_axis_mut(Axis(0), 1),
        forces.index_axis(Axis(0), 1),
        visc,
        dt,
        20,
    );

    project(vel.view_mut(), forces.view_mut(), 20);
    std::mem::swap(&mut vel, &mut forces);

    advection_positions(positions.view_mut(), vel.view(), dt);

    advect(
        Comp::X,
        vel.index_axis_mut(Axis(0), 0),
        forces.index_axis(Axis(0), 0),
        positions.view(),
    );

    advect(
        Comp::Y,
        vel.index_axis_mut(Axis(0), 1),
        forces.index_axis(Axis(0), 1),
        positions.view(),
    );
    project(vel, forces, 20);
}

/// Take a density step
/// `diff` is the diffusion constant
/// `disp` is the dispersion constant
pub fn den_step<'a>(
    mut den: ArrayViewMut2<'a, f32>,
    mut sources: ArrayViewMut2<'a, f32>,
    positions: ArrayView3<f32>,
    diff: f32,
    disp: f32,
    dt: f32,
) {
    let n = sources.shape().to_vec();
    den += &(&sources * dt);
    std::mem::swap(&mut den, &mut sources);
    diffuse(Comp::Scalar, den.view_mut(), sources.view(), diff, dt, 20);
    std::mem::swap(&mut den, &mut sources);
    advect(Comp::Scalar, den.view_mut(), sources.view(), positions);
    den -= &(disp * dt * (((n[0] - 2) * (n[1] - 2)) as f32) * &den);
}

#[cfg(test)]
mod tests {
    #[test]
    fn boundary_test() {
        use super::{set_bnd, Comp};
        use ndarray::{arr2, Array2};
        let mut my_arr = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]);
        set_bnd(Comp::X, my_arr.view_mut());
        assert_eq!(
            my_arr,
            arr2(&[
                [0.0, -1.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [0.0, 1.0, 0.0]
            ])
        );
        set_bnd(Comp::Y, my_arr.view_mut());
        assert_eq!(
            my_arr,
            arr2(&[
                [0.0, 1.0, 0.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [0.0, -1.0, 0.0]
            ])
        )
    }
    #[test]
    fn advection_positions_test() {
        use super::advection_positions;
        use ndarray::arr3;
        use ndarray::prelude::*;
        let mut p: Array3<f32> = Array3::zeros((3, 4, 2));
        let p_copy = p.clone();
        let new_vals = advection_positions(p.view_mut(), p_copy.view(), 0.2);
        assert_eq!(p_copy, p);
    }
}
