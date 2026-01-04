const EPS: f32 = 1e-7;

pub fn binary_cross_entropy(prediction: f32, target: f32) -> f32 {
    let prediction = prediction.clamp(EPS, 1.0 - EPS);
    -(target * prediction.ln() + (1.0 - target) *  (1.0 - prediction).ln())
}