use nalgebra::DVector;
use varpro::prelude::*;
use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};

pub fn pos(t: f64, c_1: f64, c_2: f64, c_3: f64, c_4: f64, c_e: f64, alpha: f64, c_c: f64) -> f64 {
    t*c_1 + t*t*c_2 + t*t*t*c_3 + t*t*t*t*c_4 + c_e * (alpha * t).exp() + c_c
}

pub fn pos_params(t: f64, params: Params) -> f64 {
    pos(t, params.c_1, params.c_2, params.c_3, params.c_4, params.c_e, params.alpha, params.c_c)
}

pub fn pos_dt(t: f64, c_1: f64, c_2: f64, c_3: f64, c_4: f64, c_e: f64, alpha: f64) -> f64 {
    c_1 + 2. * t * c_2 + 3. * t * t * c_3 + 4. * t * t * t * c_4 + c_e * alpha * (alpha * t).exp()
}

pub fn pos_1(t: &DVector<f64>) -> DVector<f64> {
    t.map(|t| t)
}

pub fn pos_2(t: &DVector<f64>) -> DVector<f64> {
    t.map(|t| t*t)
}

pub fn pos_3(t: &DVector<f64>) -> DVector<f64> {
   t.map(|t| t*t*t)
}

pub fn pos_4(t: &DVector<f64>) -> DVector<f64> {
    t.map(|t| t*t*t*t)
}

pub fn pos_e(t: &DVector<f64>, alpha: f64) -> DVector<f64> {
    t.map(|t| (alpha * t).exp())
}

pub fn pos_e_dalpha(t: &DVector<f64>, alpha: f64) -> DVector<f64> {
    t.map(|t| t * (alpha * t).exp())
}

pub fn solve(data: Vec<(f64, f64)>) -> Option<Params> {

    let (ts, ps) = data.iter().copied().unzip();
    let model = SeparableModelBuilder::<f64>::new(&["alpha"])
        // add the first exponential decay and its partial derivative to the model
        // give all parameter names that the function depends on
        // and subsequently provide the partial derivative for each parameter
        .invariant_function(pos_1)
        .invariant_function(pos_2)        
        .invariant_function(pos_3)
        .invariant_function(pos_4)
        .function(&["alpha"], pos_e)
        .partial_deriv("alpha", pos_e_dalpha)
        // add the constant as a vector of ones as an invariant function
        .invariant_function(|x|DVector::from_element(x.len(),1.))
        // the independent variable (x-axis) is the same for all basis functions
        .independent_variable(DVector::from_vec(ts))
        // the initial guess for the nonlinear parameters is alpha=1
        .initial_parameters(vec![1.])
        // build the model
        .build()
        .ok()?;
    // 2.,3: Cast the fitting problem as a nonlinear least squares minimization problem
    let problem = LevMarProblemBuilder::new(model)
        .observations(DVector::from_vec(ps))
        .build()
        .ok()?;
    // 4. Solve using the fitting problem
    let fit_result = LevMarSolver::new()
        .fit(problem)
        .expect("fit must succeed");
    // the nonlinear parameters after fitting
    // they are in the same order as the parameter names given to the model
    let alpha = fit_result.nonlinear_parameters();
    // the linear coefficients after fitting
    // they are in the same order as the basis functions that were added to the model
    let c = fit_result.linear_coefficients().unwrap();

    Some(Params { c_1: c[0], c_2: c[1], c_3: c[2], c_4: c[3], c_e: c[4], alpha: alpha[0], c_c: c[5] })
}

pub struct Params {
    pub c_1: f64,
    pub c_2: f64,
    pub c_3: f64,
    pub c_4: f64,
    pub c_e: f64,

    pub alpha: f64,

    pub c_c: f64,
}