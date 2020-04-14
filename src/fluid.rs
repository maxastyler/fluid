use crate::dynamics::{den_step, vel_step};
use gdnative::init::property::*;
use gdnative::*;
use ndarray::prelude::*;

const SHAPE: (usize, usize) = (20, 20);
const SHAPE_3: (usize, usize, usize) = (2, SHAPE.0, SHAPE.1);

/// The fluid class which contains functions for moving around a fluid
#[derive(NativeClass, Debug)]
#[inherit(Node2D)]
pub struct Fluid {
    pub vel: Array3<f32>,
    pub prev_vel: Array3<f32>,
    // new_density, prev_density, dissipation, diffusion
    pub densities: Vec<(Array2<f32>, Array2<f32>, f32, f32)>,
    pub positions: Array3<f32>,
    pub viscosity: f32,
    pub iterations: usize,
}

#[methods]
impl Fluid {
    fn _init(_owner: Node2D) -> Self {
        Fluid {
            vel: Array3::zeros(SHAPE_3),
            prev_vel: Array3::zeros(SHAPE_3),
            densities: vec![(Array2::zeros(SHAPE), Array2::zeros(SHAPE), 1.0, 1.0)],
            positions: Array3::zeros(SHAPE_3),
            viscosity: 1.0,
            iterations: 20,
        }
    }

    #[export]
    fn get_iterations(&mut self, _: Node2D) -> usize {
        self.iterations
    }

    #[export]
    fn set_iterations(&mut self, _: Node2D, i: usize) {
        self.iterations = i;
    }

    #[export]
    fn set_shape(&mut self, mut owner: Node2D, shape: (usize, usize)) {
        self.vel = Array3::zeros((2, shape.0, shape.1));
        self.prev_vel = self.vel.clone();
        for (v, pv, _, _) in self.densities.iter_mut() {
            *v = Array2::zeros(SHAPE);
            *pv = v.clone();
        }
        self.positions = self.vel.clone();
    }

    #[export]
    fn get_shape(&mut self, mut owner: Node2D) -> (usize, usize) {
        let n = self.vel.shape();
        (n[1], n[2])
    }

    #[export]
    fn get_number_of_densities(&mut self, mut owner: Node2D) -> usize {
        self.densities.len()
    }

    /// Set the number of density fields contained in this struct
    #[export]
    fn set_number_of_densities(&mut self, mut owner: Node2D, n: usize) {
        let s = self.vel.shape().to_vec();
        if self.densities.len() > n {
            self.densities.truncate(n);
        } else {
            for _ in 0..(n - self.densities.len()) {
                self.densities.push((
                    Array2::zeros((s[0], s[1])),
                    Array2::zeros((s[0], s[1])),
                    1.0,
                    1.0,
                ));
            }
        }
    }

    #[export]
    fn get_viscosity(&mut self, _: Node2D) -> f32 {
        self.viscosity
    }

    #[export]
    fn set_viscocity(&mut self, _: Node2D, v: f32) {
        self.viscosity = v;
    }

    /// Reset all fields to 0
    #[export]
    fn clear(&mut self, mut owner: Node2D) {
        self.vel.fill(0.0);
        self.prev_vel.fill(0.0);
        for (v, pv, _, _) in self.densities.iter_mut() {
            v.fill(0.0);
            pv.fill(0.0);
        }
    }

    #[export]
    fn step(&mut self, mut owner: Node2D, dt: f32) -> bool {
        vel_step(
            self.positions.view_mut(),
            self.vel.view_mut(),
            self.prev_vel.view_mut(),
            self.viscosity,
            dt,
        );
        // den_step(
        //     self.densities.view_mut(),
        //     self.prev_densities.view_mut(),
        //     self.positions.view(),
        //     self.dif,
        //     self.dis,
        //     dt,
        // );
        return true;
    }

    #[export]
    fn get_vel(&mut self, mut owner: Node2D, position: (usize, usize)) -> (f32, f32) {
        (
            self.vel[(0, position.0, position.1)],
            self.vel[(1, position.0, position.1)],
        )
    }

    #[export]
    fn set_vel(&mut self, mut owner: Node2D, position: (usize, usize), val: (f32, f32)) {
        self.vel[(0, position.0, position.1)] = val.0;
        self.vel[(1, position.0, position.1)] = val.1;
    }

    // #[export]
    // fn get_den(&mut self, mut owner: Node2D, position: (usize, usize)) -> f32 {
    //     self.densities[position]
    // }

    // #[export]
    // fn set_den(&mut self, mut owner: Node2D, position: (usize, usize), val: f32) {
    //     self.densities[position] = val;
    // }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fluid_step_test() {
        use super::Fluid;
        use super::{SHAPE, SHAPE_3};
        use crate::dynamics::{den_step, vel_step};
        use gdnative::Node2D;
        use ndarray::prelude::*;
        let mut vel = Array3::zeros(SHAPE_3);
        let mut prev_vel = Array3::zeros(SHAPE_3);
        let mut densities = Array2::zeros(SHAPE);
        let mut prev_densities = Array2::zeros(SHAPE);
        let mut positions = Array3::zeros(SHAPE_3);
        let vis = 1.0;
        let dis = 1.0;
        let dif = 1.0;
        *vel.get_mut((0, 1, 1)).unwrap() = 1.0;
        vel_step(
            positions.view_mut(),
            vel.view_mut(),
            prev_vel.view_mut(),
            vis,
            0.1,
        );
        den_step(
            densities.view_mut(),
            prev_densities.view_mut(),
            positions.view(),
            dif,
            dis,
            0.1,
        );
        println!("{:?}", vel)
    }
}
