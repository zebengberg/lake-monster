"use strict";
// import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";
// const model = await tf.loadLayersModel(
//   "/Users/robotics/projects/lake-monster/lake_monster_js/tfjs/model/model.json"
// );
const model = tf.sequential();
document.addEventListener("mousemove", (e) => {
    const x1 = e.clientX;
    const x2 = e.clientY;
    const y = 1;
    //const y = model([[x1, x2]]);
    console.log(x1, x2, y);
});
