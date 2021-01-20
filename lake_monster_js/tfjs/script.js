"use strict";
tf.loadLayersModel("./model/model.json").then((model) => {
    document.addEventListener("mousemove", (e) => {
        const x1 = e.clientX / window.innerWidth;
        const x2 = e.clientY / window.innerHeight;
        const x = tf.tensor([[x1, x2]]);
        let y = model.predict(x);
        y = y.dataSync()[0];
        console.log(x1, x2, y);
    });
});
