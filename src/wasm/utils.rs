pub fn parse_array_to_matrix(input_size: usize, input: Vec<f32>) -> Vec<Vec<f32>> {
    let mut input_parsed: Vec<Vec<f32>> = vec![];
    for i in 0..input.len() {
        if i % input_size == 0 {
            input_parsed.push(vec![input[i]]);
            continue;
        }
        input_parsed[i / input_size].push(input[i]);
    }
    input_parsed
}
