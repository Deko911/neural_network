import * as wasm from 'neural_network'
import modelData from "../models/numbers.json";

function main() {
    const GRID = 28;
    const CELL = 20;
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas!.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = '#e0e0e0';

    let drawing = false;
    function getCellFromEvent(e: MouseEvent | Touch) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / CELL);
        const y = Math.floor((e.clientY - rect.top) / CELL);
        return { x: Math.max(0, Math.min(GRID - 1, x)), y: Math.max(0, Math.min(GRID - 1, y)) };
    }

    function fillCell(x: number, y: number) {
        ctx.fillStyle = '#000';
        for (let i = 0; i <= 1; i++) {
            for (let j = 0; j <= 1; j++) {
                ctx.fillRect((x + i) * CELL, (y + j) * CELL, CELL, CELL);
            }
        }
    }

    canvas.addEventListener('mousedown', (e) => {
        drawing = true;
        const c = getCellFromEvent(e);
        fillCell(c.x, c.y);
    });
    window.addEventListener('mouseup', () => drawing = false);
    canvas.addEventListener('mousemove', (e) => {
        if (!drawing) return;
        const c = getCellFromEvent(e);
        fillCell(c.x, c.y);
    });
    
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        drawing = true;
        const touch = e.touches[0];
        const c = getCellFromEvent(touch);
        fillCell(c.x, c.y);
    }, { passive: false });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!drawing) return;
        const touch = e.touches[0];
        const c = getCellFromEvent(touch);
        fillCell(c.x, c.y);
    }, { passive: false });
    canvas.addEventListener('touchend', () => drawing = false);

    document.getElementById('clearBtn')!.addEventListener('click', () => {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    });

    
    function getGrayscaleArray() {
        const out = new Float32Array(GRID * GRID);
        for (let gy = 0; gy < GRID; gy++) {
            for (let gx = 0; gx < GRID; gx++) {
                const img = ctx.getImageData(gx * CELL, gy * CELL, CELL, CELL);
                let sum = 0;
                for (let i = 0; i < img.data.length; i += 4) {
                    const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
                    
                    const lum = (r + g + b) / 255;
                    sum += lum / 3;
                }
                const avgLum = sum / (img.data.length / 4);
                const value = 1 - avgLum;
                out[gy * GRID + gx] = value;
            }
        }
        return out;
    }

    document.getElementById('predictBtn')!.addEventListener('click', () => {
        const arr = getGrayscaleArray();
        let result = model.predict(arr);
        console.log(result);
        let max = 0.0;
        let pre = 0
        for (let i = 0; i < result.length; i++) {
            let r = result[i]
            if (r > max) {
                max = r
                pre = i
            }   
        }
        alert(`Prediction: ${pre} - Probability ${Math.trunc(max * 100)}%`)
    });

}

main();

let model = wasm.NeuralNetworkJS.load(JSON.stringify(modelData))!;