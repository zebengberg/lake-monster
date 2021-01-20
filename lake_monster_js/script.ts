const canvas = <HTMLCanvasElement>document.getElementById("canvas");
const ctx = <CanvasRenderingContext2D>canvas.getContext("2d");
const slider = <HTMLInputElement>document.getElementById("slider");
const label = <HTMLLabelElement>document.getElementById("label");
const checkbox = <HTMLInputElement>document.getElementById("checkbox");

const size = Math.min(window.innerWidth, window.innerHeight) - 100;
canvas.height = size;
canvas.width = size;
const radius = 0.45 * size;
const stepSize = 0.005;

let monsterSpeed = Number(slider.value);
label.innerText = "monster speed: " + slider.value;
slider.onchange = () => {
  monsterSpeed = Number(slider.value);
  label.innerText = "monster speed: " + slider.value;
};

class Point {
  x: number;
  y: number;
  color: string | null;

  constructor(x: number, y: number, color: string | null) {
    this.x = x;
    this.y = y;
    this.color = color;
  }

  reset(x: number, y: number) {
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
    const y = size / 2 + radius * this.y;
    const r = 10;

    ctx.beginPath();
    if (this.color !== null) {
      ctx.fillStyle = this.color;
    }
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
  }

  moveToTarget(target: Point) {
    let dx = target.x - this.x;
    let dy = target.y - this.y;
    const d = new Point(dx, dy, null);
    if (stepSize > d.norm) {
      this.x = target.x;
      this.y = target.y;
    } else {
      dx *= stepSize / d.norm;
      dy *= stepSize / d.norm;
      this.x += dx;
      this.y += dy;
    }
  }

  moveAlongArc(targetAngle: number) {
    const dAngle = angleDiff(targetAngle, this.angle);
    const arcSize = monsterSpeed * stepSize;
    if (arcSize > Math.abs(dAngle)) {
      this.x = Math.cos(targetAngle);
      this.y = Math.sin(targetAngle);
    } else {
      const newAngle = this.angle + Math.sign(dAngle) * arcSize;
      this.x = Math.cos(newAngle);
      this.y = Math.sin(newAngle);
    }
  }
}

function drawLake() {
  ctx.clearRect(0, 0, size, size);
  ctx.beginPath();
  ctx.fillStyle = "blue";
  ctx.arc(size / 2, size / 2, radius, 0, 2 * Math.PI);
  ctx.closePath();
  ctx.fill();
  if (checkbox.checked) {
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, radius / monsterSpeed, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function angleDiff(angle1: number, angle2: number) {
  let dAngle = angle1 - angle2;
  while (dAngle < -Math.PI) {
    dAngle += 2 * Math.PI;
  }
  while (dAngle > Math.PI) {
    dAngle -= 2 * Math.PI;
  }
  return dAngle;
}

function determineWinner(p1: Point, p2: Point) {
  const dAngle = angleDiff(p1.angle, p2.angle);
  return Math.abs(dAngle) < 0.000001 ? "monster wins!" : "you win!";
}

const agent = new Point(0, 0, "red");
const monster = new Point(1, 0, "lime");
const mouse = new Point(0, 0, null);

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = (e.clientX - rect.left - size / 2) / radius;
  mouse.y = (e.clientY - rect.top - size / 2) / radius;
});

let gameOver = false;
document.addEventListener("keydown", (e) => {
  if (e.key == " ") {
    if (gameOver) {
      agent.reset(0, 0);
      monster.reset(1, 0);
      gameOver = false;
    }
  }
});

function run() {
  drawLake();
  agent.draw();
  monster.draw();

  if (agent.norm > 1) {
    gameOver = true;
    const winner = determineWinner(agent, monster);
    ctx.font = "small-caps bold 24px arial";
    ctx.textAlign = "center";
    ctx.fillStyle = "white";
    ctx.fillText(winner, size / 2, size / 2 - 15);
    ctx.fillText("press space to restart", size / 2, size / 2 + 15);
  }
  if (!gameOver) {
    agent.moveToTarget(mouse);
    monster.moveAlongArc(agent.angle);
  }
}

setInterval(run, 50);
