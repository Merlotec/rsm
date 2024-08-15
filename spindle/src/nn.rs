extern crate tensorflow;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, Operation};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn build_model(graph: &mut Graph, input: Operation, num_classes: i32) -> (Operation, Operation) {
    // First dense layer
    let mut weights_1 = Tensor::new(&[3, 64]).with_values(&[0.1; 3 * 64]).unwrap();
    let mut biases_1 = Tensor::new(&[64]).with_values(&[0.1; 64]).unwrap();
    let layer_1 = graph.add(
        tensorflow::ops::add(
            graph,
            tensorflow::ops::mat_mul(graph, input, weights_1, false, false).unwrap(),
            biases_1,
        )
        .unwrap(),
    )
    .unwrap();
    
    let relu_1 = tensorflow::ops::relu(graph, layer_1).unwrap();

    // Second dense layer
    let mut weights_2 = Tensor::new(&[64, 32]).with_values(&[0.1; 64 * 32]).unwrap();
    let mut biases_2 = Tensor::new(&[32]).with_values(&[0.1; 32]).unwrap();
    
    let layer_2 = graph.add(
        tensorflow::ops::add(
            graph,
            tensorflow::ops::mat_mul(graph, relu_1, weights_2, false, false).unwrap(),
            biases_2,
        )
        .unwrap(),
    )
    .unwrap();

    let relu_2 = tensorflow::ops::relu(graph, layer_2).unwrap();

    // Output layer
    let mut weights_3 = Tensor::new(&[32, num_classes]).with_values(&[0.1; 32 * num_classes as usize]).unwrap();
    let mut biases_3 = Tensor::new(&[num_classes]).with_values(&[0.1; num_classes as usize]).unwrap();

    let logits = graph.add(
        tensorflow::ops::add(
            graph,
            tensorflow::ops::mat_mul(graph, relu_2, weights_3, false, false).unwrap(),
            biases_3,
        )
        .unwrap(),
    )
    .unwrap();

    let softmax = tensorflow::ops::softmax(graph, logits).unwrap();

    (softmax, logits)
}

fn main() {
    // Define hyperparameters
    let input_shape = (3,);  // (velocity1, velocity2, position2)
    let num_classes = 10;

    // Create the graph and session
    let mut graph = Graph::new();
    let mut session = Session::new(&SessionOptions::new(), &graph).unwrap();

    // Define placeholders for input data and labels
    let input = graph.placeholder(input_shape, tensorflow::DataType::Float);
    let labels = graph.placeholder((1,), tensorflow::DataType::Int32);

    // Build the model
    let (softmax, logits) = build_model(&mut graph, input, num_classes);

    // Define loss and optimizer
    let loss = tensorflow::ops::sparse_softmax_cross_entropy_with_logits(graph, logits, labels).unwrap();
    let optimizer = tensorflow::ops::adam_optimizer(graph, 0.001).minimize(loss).unwrap();

    // Generate random training data (replace this with your actual data)
    let mut rng = rand::thread_rng();
    let x_train: Array2<f32> = Array2::random_using((1000, 3), Uniform::new(0.0, 1.0), &mut rng);
    let y_train: Vec<i32> = (0..1000).map(|_| rng.gen_range(0..num_classes)).collect();

    // Convert the training data to TensorFlow tensors
    let x_train_tensor = Tensor::new(&[1000, 3]).with_values(x_train.as_slice().unwrap()).unwrap();
    let y_train_tensor = Tensor::new(&[1000]).with_values(&y_train).unwrap();

    // Training loop
    for epoch in 0..10 {
        let mut args = SessionRunArgs::new();
        args.add_feed(&input, 0, &x_train_tensor);
        args.add_feed(&labels, 0, &y_train_tensor);
        args.add_target(&optimizer);

        session.run(&mut args).unwrap();
        println!("Epoch {} completed", epoch + 1);
    }

    // Example prediction
    let mut x_test_tensor = Tensor::new(&[1, 3]).with_values(&[0.5, 0.2, 0.1]).unwrap();
    let mut args = SessionRunArgs::new();
    let softmax_fetch = args.request_fetch(&softmax, 0);
    args.add_feed(&input, 0, &x_test_tensor);

    session.run(&mut args).unwrap();

    let softmax_output: Tensor<f32> = args.fetch(softmax_fetch).unwrap();
    let predicted_class = softmax_output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    println!("Predicted class: {}", predicted_class + 1);  // +1 to match the class label
}
