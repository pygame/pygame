window.my_data = {
    level: 1
};
//电子邮件puhalskijsemen@gmail.com
//源码网站 开vpn全局模式打开 http://web3incubators.com/
//电报https://t.me/gamecode999
//网页客服 http://web3incubators.com/kefu.html
cc.Class({
    extends: cc.Component,

    properties: {
        gameLayout: cc.Node,
        item11: cc.Prefab,
        xiaochuAniNode: cc.Node,
        passLabel: cc.RichText,
        jiheNode: cc.Node,
        tooltips: cc.Node
    },

    // LIFE-CYCLE CALLBACKS:

    onLoad() {
        this.init_properties(this)
    },
    init_properties(t) {
        t.MajiangPop = [];
        t.level = 1;
        t.levelData = [{
            jieduan: [1, 9],
            huase: [3, 3],
            Max: [3, 6],
            lie: 5,
            passNumber: 10
        }, {
            jieduan: [9, 18, 27, 54, 81],
            huase: [3, 3, 3, 3, 3],
            Max: [5, 6, 7, 8, 9],
            lie: 9,
            passNumber: 175
        }];
        t.passNumber = 0;
        t.nowNumber = 0;
        t.upStarty = 250;
        t.upYjianju = 117;//和 item11 高一样 ，因为放大1.3 倍所以得 90*1.3
        t.lie = 2;
        t.hang = 2;
        t.startX = 0;
        t.jianjuX = 84.5;//和 item11 宽一样 ，因为放大1.3 倍所以得 65*1.3
        t.jianjuY = 117;
        t.jjx = 0;
        t.downStartY = -200;
        t.UpNode = [];
        t.MjNode = [];
        t.nowUpIndex = 0;
        t.jiaohuanIng = !1;
        t.huhuanTween = !1;
        t.huhuanche = [null, null];
        t.daluanIng = !1
    },
    start() {
        this.level = my_data.level;
        this.initMap();
    },
    initMap() {
        
        var self = this;
        this.gameLayout.removeAllChildren();
        var levelData = this.levelData[this.level - 1];
        this.passNumber = levelData.passNumber;
        this.nowNumber = 0;
        var arr_single = [], arr_double = [];
        for (var i = 0; i &lt; levelData.jieduan.length; i++) {
            for (var a = levelData.jieduan[i], r = 0; r &lt; a; r++) {
                var c = i,
                    l = Math.floor(Math.random() * levelData.huase[i]),
                    u = Math.floor(Math.random() * (levelData.Max[i] - 2)) + 1,
                    p = Math.random(),
                    m = p &lt; .5 ? u + 1 : u,
                    y = p &lt; .5 ? u - 1 : u;
                if (u == m && u &lt;= 7 && Math.random() &lt; .5) {
                    u--;
                    m--;
                    y--;
                    l = 3
                }

                var single = {
                        huaseId: l,
                        idData: [y],
                        cengid: c
                    },
                    double = {
                        huaseId: l,
                        idData: [u, m],
                        cengid: c
                    };
                arr_single.push(single);
                arr_double.push(double);
            }
        }
        this.upStarty = (this.node.height / 2 - 190 - 240 + 5) / this.gameLayout.scale;

        var s_y = this.upStarty,
            spacing = this.upYjianju,
            j = arr_double;
        //上方 预制
        for (var w = 0; w &lt; j.length; w++) {
            var b = j[w],
                item = cc.instantiate(this.item11);
            item.getComponent("item11").setId(b.idData, b.huaseId, false);
            item.getComponent("item11").cengid = b.cengid;
            item.parent = this.gameLayout;
            item.x = 0;
            item.y = s_y;
            s_y += spacing;
            this.UpNode.push(item);
        }

        this.lie = levelData.lie;
        if (levelData.lie % 2 == 0) {
            this.startX = -levelData.lie / 2 * (this.jianjuX + this.jjx) + (this.jianjuX + this.jjx)
        } else {
            this.startX = -(levelData.lie - 1) / 2 * (this.jianjuX + this.jjx) + (this.jianjuX + this.jjx) - (this.jianjuX - this.jjx);
        }

        this.downStartY = (-this.node.height / 2 + 620) / this.gameLayout.scale;

        this.hang = arr_double.length / this.lie;
        var x = this.downStartY,
            count = 0, data;
        //下方 预制
        for (r = 0; r &lt; this.hang; r++) {
            for (var xx = this.startX, I = 0; I &lt; this.lie; I++) {
                data = arr_single[count];
                if (data) {
                    item = cc.instantiate(this.item11);
                    item.getComponent("item11").setId(data.idData, data.huaseId, true);
                    item.getComponent("item11").cengid = data.cengid;
                    item.parent = this.gameLayout;
                    item.x = xx;
                    item.y = x;
                    item.on(cc.Node.EventType.TOUCH_START, this.touchEndItem, this);
                    this.MjNode.push(item);
                    xx += this.jianjuX;
                    count++
                }

            }

            x -= this.jianjuY;
        }
        //如果 this.gameLayout 的子节点还要更多，就需要自己U化下

        this.nowUpIndex = 0;
        this.daluan();
    },
    daluan(isTween) {
        if (!this.daluanIng) {
    
