<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Spartan Campus Explorer - MSU</title>
    <style>
        body { margin:0; background:#000; display:flex; justify-content:center; align-items:center; height:100vh; font-family:Arial; }
        canvas { image-rendering: pixelated; border: 4px solid #8B0000; max-width:100%; }
        #instructions { position:absolute; top:10px; left:10px; color:white; background:rgba(0,0,0,0.7); padding:10px; border-radius:8px; font-size:14px; }
    </style>
</head>
<body>
    <canvas id="game" width="800" height="600"></canvas>
    <div id="instructions">
        Arrow keys or WASD to move<br>
        Collect items • Visit stadiums & Crunchy's!<br>
        <span style="color:#FFD700">Go Green!</span>
    </div>

    <script>
        const canvas = document.getElementById('game');
        const ctx = canvas.getContext('2d');
        
        const WIDTH = 800;
        const HEIGHT = 600;
        let player = { x: 400, y: 300, size: 30, speed: 5 };
        
        const buildings = [
            {x:80, y:80, w:130, h:90},
            {x:280, y:40, w:140, h:110},
            {x:520, y:180, w:110, h:130},
            {x:120, y:320, w:90, h:140}
        ];
        
        const venues = [
            {x:450, y:30, w:180, h:120, name: "SPARTAN STADIUM", color: "#006400"},
            {x:50, y:480, w:140, h:90, name: "BRESLIN CENTER", color: "#8B0000"},
            {x:580, y:420, w:130, h:100, name: "MUNN ARENA", color: "#8B0000"},
            {x:320, y:380, w:110, h:70, name: "CRUNCHY'S", color: "#FF4500"}
        ];
        
        let collectibles = [];
        for(let i = 0; i < 12; i++) {
            collectibles.push({
                x: Math.random() * (WIDTH - 60) + 30,
                y: Math.random() * (HEIGHT - 60) + 30,
                type: ['book','helmet','apple','ball'][Math.floor(Math.random()*4)],
                size: 22
            });
        }
        
        let score = 0;
        let collected = 0;
        const total = collectibles.length;
        
        const keys = {};
        
        window.addEventListener('keydown', e => keys[e.key.toLowerCase()] = true);
        window.addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);
        
        function update() {
            if (keys['arrowleft'] || keys['a']) player.x -= player.speed;
            if (keys['arrowright'] || keys['d']) player.x += player.speed;
            if (keys['arrowup'] || keys['w']) player.y -= player.speed;
            if (keys['arrowdown'] || keys['s']) player.y += player.speed;
            
            player.x = Math.max(0, Math.min(WIDTH - player.size, player.x));
            player.y = Math.max(0, Math.min(HEIGHT - player.size, player.y));
            
            for(let i = collectibles.length - 1; i >= 0; i--) {
                const c = collectibles[i];
                if (Math.hypot(player.x + player.size/2 - c.x, player.y + player.size/2 - c.y) < 25) {
                    collectibles.splice(i, 1);
                    score += 10;
                    collected++;
                }
            }
        }
        
        function draw() {
            ctx.fillStyle = '#228B22';
            ctx.fillRect(0, 0, WIDTH, HEIGHT);
            
            ctx.fillStyle = '#808080';
            ctx.fillRect(0, 240, WIDTH, 110);
            ctx.fillRect(340, 0, 90, HEIGHT);
            
            ctx.fillStyle = '#4682B4';
            buildings.forEach(b => {
                ctx.fillRect(b.x, b.y, b.w, b.h);
                ctx.strokeStyle = '#8B0000';
                ctx.lineWidth = 5;
                ctx.strokeRect(b.x, b.y, b.w, b.h);
            });
            
            venues.forEach(v => {
                ctx.fillStyle = v.color;
                ctx.fillRect(v.x, v.y, v.w, v.h);
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 6;
                ctx.strokeRect(v.x, v.y, v.w, v.h);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 14px Arial';
                ctx.fillText(v.name, v.x + 10, v.y + 22);
            });
            
            ctx.fillStyle = '#FFD700';
            ctx.beginPath();
            ctx.arc(player.x + player.size/2, player.y + player.size/2, player.size/2, 0, Math.PI*2);
            ctx.fill();
            ctx.fillStyle = '#8B0000';
            ctx.fillRect(player.x + 8, player.y + 12, player.size - 16, 14);
            
            collectibles.forEach(c => {
                if (c.type === 'book') {
                    ctx.fillStyle = 'white';
                    ctx.fillRect(c.x, c.y, 22, 28);
                } else if (c.type === 'helmet') {
                    ctx.fillStyle = '#8B0000';
                    ctx.beginPath();
                    ctx.arc(c.x+11, c.y+11, 13, 0, Math.PI*2);
                    ctx.fill();
                } else if (c.type === 'ball') {
                    ctx.fillStyle = '#FFD700';
                    ctx.beginPath();
                    ctx.arc(c.x+11, c.y+11, 11, 0, Math.PI*2);
                    ctx.fill();
                } else {
                    ctx.fillStyle = '#FF0000';
                    ctx.beginPath();
                    ctx.arc(c.x+11, c.y+11, 12, 0, Math.PI*2);
                    ctx.fill();
                }
            });
            
            ctx.fillStyle = 'white';
            ctx.font = 'bold 24px Arial';
            ctx.fillText(`Score: ${score}   Items: ${collected}/${total}`, 20, 35);
            
            if (collected >= total) {
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                ctx.fillRect(0, 0, WIDTH, HEIGHT);
                ctx.fillStyle = '#FFD700';
                ctx.font = 'bold 32px Arial';
                ctx.fillText("🏆 MSU CAMPUS CONQUERED!", 110, 280);
                ctx.font = 'bold 26px Arial';
                ctx.fillText("Go Green! 🟢", 310, 330);
            }
        }
        
        function gameLoop() {
            update();
            draw();
            requestAnimationFrame(gameLoop);
        }
        
        gameLoop();
    </script>
</body>
</html>
