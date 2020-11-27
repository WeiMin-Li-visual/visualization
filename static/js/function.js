init_graph = function(graph, flag=true){
    graph = JSON.parse(graph);

    //给节点的属性进行操作
    graph.nodes.forEach(function (node) {
        node.symbolSize = 15;
        // node.value = node.symbolSize;
        if(flag == true)
            node.category = node.attributes.modularity_class;
        node.draggable = true;
    });
    return graph;
};

init_option = function(graph){
    var option;
    var categories = [];//设置节点属性
    categories[0] = {
            name: '未激活节点',
            itemStyle: {
                color: '#2f4554',
                opacity: 0.9, //不透明度
            }
        }
        categories[1] = {
            name: '激活节点',
            itemStyle: {
                color: '#009688',
                opacity: 1, //不透明度
            },
            label: {
                fontSize: 20,
            },

        };
    //设置要生成的图的相关属性
    option = {
        title: {
            text: '',
            subtext: '',
            top: 'top',
            left: 'left'
        },
        tooltip: {},
        legend: [{
            // selectedMode: 'single',
            data: categories.map(function (a) {
                return a.name;
            })
        }],
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [
            {
                name: 'Les Miserables',
                type: 'graph',
                layout: 'none',
                //下面三项分别设置节点数据，边数据，种类数据
                data: graph.nodes,
                links: graph.links,
                categories: categories,
                roam: true,
                animation: false,
                focusNodeAdjacency: true,
                itemStyle: {
                    normal: {
                        borderColor: '#fff',
                        borderWidth: 1,
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.3)'
                    }
                },
                label: {//设置节点标签
                    position: 'right',
                    formatter: '{b}'
                },
                lineStyle: {//边的种类
                    color: 'rgba(0, 0, 0, 0.7)',
                    curveness: 0.3
                },
                emphasis: {
                    lineStyle: {
                        width: 10
                    }
                },
                force: {
                    repulsion: 270
                }
            }
        ]
    };
    return option;
}
function getUrlParam(name) {
            var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)"); //构造一个含有目标参数的正则表达式对象
            var r = window.location.search.substr(1).match(reg);  //匹配目标参数
            if (r != null) return unescape(r[2]);
            return null; //返回参数值
        }
redirection = function()  {
    let userName = getUrlParam("userName");
    $.ajax({
        type: "POST",
        url: "/fun",
        data: {
            "userName": userName
        },
        success: function(data) {
            if(data.userInfo == 1)
                $(window).attr('location','/login');
        },
        error: function() {
            console.log('error');
        }
    })
}
