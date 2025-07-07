<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>合成大西瓜</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        #gameCanvas {
            background-color: #a0d8a0;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }
        #score {
            position: absolute;
            top: 20px;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="score">分数: 0</div>
    <canvas id="gameCanvas" width="400" height="600"></canvas>

    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const scoreElement = document.getElementById('score');
        
        // 游戏状态
        let score = 0;
        let fruits = [];
        let currentFruit = null;
        let gameOver = false;
        let nextFruitType = null;
        
        // 水果类型和属性
        const fruitTypes = [
            { name: '樱桃', radius: 20, color: '#ff0000', score: 1 },
            { name: '草莓', radius: 25, color: '#ff5050', score: 2 },
            { name: '葡萄', radius: 30, color: '#800080', score: 3 },
            { name: '橙子', radius: 35, color: '#ffa500', score: 4 },
            { name: '柠檬', radius: 40, color: '#ffff00', score: 5 },
            { name: '猕猴桃', radius: 45, color: '#90ee90', score: 6 },
            { name: '番茄', radius: 50, color: '#ff6347', score: 7 },
            { name: '桃子', radius: 55, color: '#ffd700', score: 8 },
            { name: '菠萝', radius: 60, color: '#ffd700', score: 9 },
            { name: '椰子', radius: 65, color: '#f5f5dc', score: 10 },
            { name: '西瓜', radius: 70, color: '#008000', score: 11 }
        ];
        
        // 初始化游戏
        function init() {
            fruits = [];
            score = 0;
            gameOver = false;
            scoreElement.textContent = `分数: ${score}`;
            spawnNextFruit();
            requestAnimationFrame(gameLoop);
        }
        
        // 游戏主循环
        function gameLoop() {
            if (gameOver) return;
            
            update();
            render();
            
            requestAnimationFrame(gameLoop);
        }
        
        // 更新游戏状态
        function update() {
            // 更新当前下落的水果
            if (currentFruit) {
                currentFruit.y += currentFruit.velocityY;
                currentFruit.velocityY += 0.2; // 重力加速度
                
                // 检测与底部或其他水果的碰撞
                if (checkCollision(currentFruit)) {
                    currentFruit.velocityY = 0;
                    fruits.push(currentFruit);
                    checkMerge(currentFruit);
                    spawnNextFruit();
                }
            }
            
            // 检查游戏结束条件
            if (fruits.some(fruit => fruit.y - fruit.radius <= 0)) {
                gameOver = true;
                alert(`游戏结束! 你的分数是: ${score}`);
                init();
            }
        }
        
        // 渲染游戏
        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 绘制水果
            fruits.forEach(fruit => {
                drawFruit(fruit);
            });
            
            // 绘制当前下落的水果
            if (currentFruit) {
                drawFruit(currentFruit);
            }
            
            // 绘制下一个水果的预览
            if (nextFruitType !== null) {
                ctx.fillStyle = '#333';
                ctx.font = '16px Arial';
                ctx.fillText('下一个:', 10, 30);
                
                const previewX = 80;
                const previewY = 30;
                const previewRadius = 15;
                
                ctx.beginPath();
                ctx.arc(previewX, previewY, previewRadius, 0, Math.PI * 2);
                ctx.fillStyle = fruitTypes[nextFruitType].color;
                ctx.fill();
                ctx.stroke();
            }
        }
        
        // 绘制水果
        function drawFruit(fruit) {
            ctx.beginPath();
            ctx.arc(fruit.x, fruit.y, fruit.radius, 0, Math.PI * 2);
            ctx.fillStyle = fruit.color;
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.stroke();
            
            // 绘制水果高光
            ctx.beginPath();
            ctx.arc(fruit.x - fruit.radius/3, fruit.y - fruit.radius/3, fruit.radius/4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.fill();
        }
        
        // 生成下一个水果
        function spawnNextFruit() {
            if (nextFruitType === null) {
                nextFruitType = Math.floor(Math.random() * 3); // 初始只生成小水果
            }
            
            const type = nextFruitType;
    
