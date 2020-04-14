use crate::dynamics::{den_step, vel_step};
use gdnative::init::property::*;
use gdnative::*;
use ndarray::prelude::*;

const SHAPE: (usize, usize) = (32, 32);
const SHAPE_3: (usize, usize, usize) = (2, SHAPE.0, SHAPE.1);

#[derive(NativeClass, Debug)]
#[inherit(Node2D)]
pub struct Fluid {
    pub vel: Array3<f32>,
    pub prev_vel: Array3<f32>,
    pub densities: Array2<f32>,
    pub prev_densities: Array2<f32>,
    pub positions: Array3<f32>,
    pub vis: f32,
    pub dis: f32,
    pub dif: f32,
}

#[methods]
impl Fluid {
    fn _init(_owner: Node2D) -> Self {
        Fluid {
            vel: Array3::zeros(SHAPE_3),
            prev_vel: Array3::zeros(SHAPE_3),
            densities: Array2::zeros(SHAPE),
            prev_densities: Array2::zeros(SHAPE),
            positions: Array3::zeros(SHAPE_3),
            vis: 1.0,
            dis: 1.0,
            dif: 1.0,
        }
    }

    #[export]
    fn step(&mut self, mut owner: Node2D, dt: f32) {
        vel_step(
            self.positions.view_mut(),
            self.vel.view_mut(),
            self.prev_vel.view_mut(),
            self.vis,
            dt,
        );
        den_step(
            self.densities.view_mut(),
            self.prev_densities.view_mut(),
            self.positions.view(),
            self.dif,
            self.dis,
            dt,
        );
    }

    #[export]
    fn get_vel(&mut self, mut owner: Node2D, position: (usize, usize)) -> (f32, f32) {
        (
            self.positions[(0, position.0, position.1)],
            self.positions[(1, position.0, position.1)],
        )
    }

    #[export]
    fn set_vel(&mut self, mut owner: Node2D, position: (usize, usize), val: (f32, f32)) {
        self.positions[(0, position.0, position.1)] = val.0;
        self.positions[(1, position.0, position.1)] = val.1;
    }

    #[export]
    fn get_den(&mut self, mut owner: Node2D, position: (usize, usize)) -> f32 {
        self.densities[position]
    }

    #[export]
    fn set_den(&mut self, mut owner: Node2D, position: (usize, usize), val: f32) {
        self.densities[position] = val;
    }
}
