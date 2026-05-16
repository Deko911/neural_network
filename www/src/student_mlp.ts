import * as wasm from 'neural_network'
import modelData from "../models/students.json";

let data = new Float32Array([
    2.0, 60.0, 50.0,
    4.0, 70.0, 65.0,
    6.0, 80.0, 70.0,
    8.0, 90.0, 85.0,
    1.0, 50.0, 40.0,
    3.0, 65.0, 55.0,
    7.0, 85.0, 80.0,
    9.0, 95.0, 90.0,
]);

let targets = new Float32Array([0, 0, 1, 1, 0, 0, 1, 1]);

let model = wasm.NeuralNetworkJS.load(JSON.stringify(modelData))!;
console.log(model.evaluate(data, targets));

let studyHours = document.getElementById("study_hours")! as HTMLInputElement;
let asistence = document.getElementById("asistence")! as HTMLInputElement;
let activities = document.getElementById("activities")! as HTMLInputElement;

function predictStudent() {
    let student = new Float32Array([Number(studyHours.value), Number(asistence.value), Number(activities.value)])
    let option = model.predict(student)[0]
    
    alert(option > 0.5 ? "approve" : "disapprove")
}

document.getElementById("predict")!.onclick = predictStudent