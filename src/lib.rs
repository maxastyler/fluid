extern crate gdnative;
extern crate euclid;
extern crate ndarray;

pub mod dynamics;
pub mod fluid;

use gdnative::*;
use crate::fluid::Fluid;

// Function that registers all exposed classes to Godot
fn init(handle: gdnative::init::InitHandle) {
    handle.add_class::<Fluid>();
}

// macros that create the entry-points of the dynamic library.
godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
