"use strict";
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("slider");
const label = document.getElementById("label");
const cirCheckbox = document.getElementById("cir-checkbox");
const aiCheckbox = document.getElementById("ai-checkbox");
const pathCheckbox = document.getElementById("path-checkbox");
const size = Math.min(window.innerWidth, window.innerHeight) - 100;
canvas.height = size;
canvas.width = size;
const radius = 0.45 * size;
const stepSize = 0.02;
let monsterSpeed = Number(slider.value);
label.innerText = "monster speed: " + slider.value;
slider.oninput = () => {
    monsterSpeed = Number(slider.value);
    label.innerText = "monster speed: " + slider.value;
};
class Point {
    constructor(x, y, color = null) {
        this.x = x;
        this.y = y;
        this.color = color;
    }
    reset(x, y) {
        this.x = x;
        this.y = y;
    }
    get angle() {
        return Math.atan2(this.y, this.x);
    }
    get norm() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }
    draw() {
        const x = size / 2 + radius * this.x;
        const y = size / 2 - radius * this.y;
        const r = 10;
        ctx.beginPath();
        if (this.color !== null) {
            ctx.fillStyle = this.color;
        }
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.closePath();
        ctx.fill();
    }
    moveToTarget(target) {
        let dx = target.x - this.x;
        let dy = target.y - this.y;
        const d = new Point(dx, dy);
        if (stepSize > d.norm) {
            this.x = target.x;
            this.y = target.y;
        }
        else {
            dx *= stepSize / d.norm;
            dy *= stepSize / d.norm;
            this.x += dx;
            this.y += dy;
        }
    }
    moveAlongArc(targetAngle) {
        const dAngle = angleDiff(targetAngle, this.angle);
        const arcSize = monsterSpeed * stepSize;
        if (arcSize > Math.abs(dAngle)) {
            this.x = Math.cos(targetAngle);
            this.y = Math.sin(targetAngle);
        }
        else {
            const newAngle = this.angle + Math.sign(dAngle) * arcSize;
            this.x = Math.cos(newAngle);
            this.y = Math.sin(newAngle);
        }
    }
}
class Path {
    constructor(color) {
        this.path = [];
        this.color = color;
    }
    append(p) {
        // cloning p
        const [x, y] = [p.x, p.y];
        this.path.push(new Point(x, y));
    }
    draw() {
        if (this.path.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(size / 2 + radius * this.path[0].x, size / 2 - radius * this.path[0].y);
            for (let p of this.path) {
                ctx.lineTo(size / 2 + radius * p.x, size / 2 - radius * p.y);
            }
            ctx.lineWidth = 3;
            ctx.strokeStyle = this.color;
            ctx.stroke();
        }
    }
    reset() {
        this.path = [];
    }
}
function drawLake() {
    ctx.clearRect(0, 0, size, size);
    ctx.beginPath();
    ctx.fillStyle = "blue";
    ctx.arc(size / 2, size / 2, radius, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
    if (cirCheckbox.checked) {
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, radius / monsterSpeed, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "black";
        ctx.stroke();
    }
}
function angleDiff(angle1, angle2) {
    let dAngle = angle1 - angle2;
    while (dAngle < -Math.PI) {
        dAngle += 2 * Math.PI;
    }
    while (dAngle > Math.PI) {
        dAngle -= 2 * Math.PI;
    }
    return dAngle;
}
function drawWinner(pAgent, pMonster) {
    const dAngle = angleDiff(pAgent.angle, pMonster.angle);
    const winner = Math.abs(dAngle) < 0.000001 ? "monster wins!" : "you win!";
    ctx.font = "small-caps bold 24px arial";
    ctx.textAlign = "center";
    ctx.fillStyle = "white";
    ctx.fillText(winner, size / 2, size / 2 - 15);
    ctx.fillText("press space to restart", size / 2, size / 2 + 15);
}
function getState(pAgent, pMonster, pPath) {
    return [
        stepSize,
        monsterSpeed,
        (pPath.path.length * stepSize) / 3.0,
        pAgent.norm,
        angleDiff(pMonster.angle, pAgent.angle),
    ];
}
function predToDirection(pred, pAgent) {
    console.log(pred);
    let max = pred[0];
    let maxIndex = 0;
    for (let i = 1; i < pred.length; i++) {
        if (pred[i] > max) {
            max = pred[i];
            maxIndex = i;
        }
    }
    const theta = pAgent.angle + (2 * Math.PI * maxIndex) / pred.length;
    const target = new Point(Math.cos(theta), Math.sin(theta));
    console.log(maxIndex, pred.length);
    console.log(pAgent.angle);
    console.log(theta);
    console.log(target);
    return target;
}
function play(pAgent, pMonster, pPath, aiMove) {
    drawLake();
    pAgent.draw();
    pMonster.draw();
    if (pathCheckbox.checked) {
        pPath.draw();
    }
    let state = getState(pAgent, pMonster, pPath);
    console.log("#######################");
    console.log(agent);
    console.log(monster);
    console.log(state);
    if (pAgent.norm >= 1) {
        gameOver = true;
        drawWinner(pAgent, pMonster);
    }
    if (!gameOver) {
        if (aiCheckbox.checked) {
            state = getState(pAgent, pMonster, pPath);
            const target = aiMove(state);
            pAgent.moveToTarget(target);
        }
        else {
            pAgent.moveToTarget(mouse);
        }
        pMonster.moveAlongArc(pAgent.angle);
        pPath.append(pAgent);
    }
    state = getState(pAgent, pMonster, pPath);
    console.log(state);
}
const agent = new Point(0, 0, "red");
const monster = new Point(1, 0, "lime");
const mouse = new Point(0, 0);
const path = new Path(agent.color);
path.append(agent);
canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = (e.clientX - rect.left - size / 2) / radius;
    mouse.y = -(e.clientY - rect.top - size / 2) / radius;
});
let gameOver = false;
document.addEventListener("keydown", (e) => {
    if (e.key == " ") {
        if (gameOver) {
            agent.reset(0, 0);
            monster.reset(1, 0);
            path.reset();
            gameOver = false;
        }
    }
});
tf.loadGraphModel("./saved_model/model.json").then((model) => {
    const aiMove = (state) => {
        const x = tf.tensor([state]);
        let y = model.predict(x);
        y = y.dataSync();
        const target = predToDirection(y, agent);
        return target;
    };
    const update = () => play(agent, monster, path, aiMove);
    // document.addEventListener("keydown", (e) => {
    //   if (e.key == "a") {
    //     update();
    //   }
    // });
    setInterval(update, 50);
});
