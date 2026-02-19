(function() {
  // 1. 实例化对象
  var myChart = echarts.init(document.querySelector(".bar .chart"));
  var myColor = [
    "#006cff",
    "#60cda0",
    "#ed8884",
    "#ff9f7f",
    "#0096ff",
    "#9fe6b8",
    "#32c5e9",
    "#1d9dff"
  ];
  // 2. 配置项
  var option = {
      tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c}  ({d}%)'
      },
      legend: {
          bottom: 0,
          data: ['< -5%', '-5% ~ 0%', '0% ~ 5%', '> 5%'], // 这里改成涨跌幅区间
          textStyle: {
            color: "#ffffff",
            fontSize: "12"
          }
      },
      toolbox: {
          show: false,
          orient: 'vertical',
          feature: {
              dataZoom: {
                  yAxisIndex: 'none'
              },
              dataView: {readOnly: false},
              magicType: {type: ['line', 'bar']},
              restore: {},
              saveAsImage: {}
          }
      },
      series: [
        {
            name: 'Stock Price Change Distribution',
            type: 'pie',
            center: ['50%', '40%'],
            radius: ['40%', '60%'],
            roseType: false,
            avoidLabelOverlap: false,
            label: {
                show: true,
                fontSize: 12,
                position: 'left',
                formatter: '{b}: {c}  ({d}%)'
            },
            labelLine: {
                length: 6,
                length2: 8,
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: '20',
                    fontWeight: 'bold'
                }
            },
            itemStyle: {
                color: function (params) {
                    var brightColors = [
                        "#73d13d", "#ebffa7","#ff4d4f",
                        "#36cfc9", "#40a9ff", "#597ef7", "#9254de"
                    ];
                    return brightColors[params.dataIndex % brightColors.length];
                }
            },
            data: [] // 由 AJAX 动态填充
        }
    ]
  };

  // 3. 渲染图表
  myChart.setOption(option);
  window.addEventListener("resize", function() {
    myChart.resize();
  });

  // 4. 更新数据
  function update_data1() {
      $.ajax({
          url: "/data?dname=data1",
          timeout: 10000, // 超时时间设置，单位毫秒
          success: function (data) {
              console.log("股票涨跌幅数据:", data);

              // 转换数据格式
              let formattedData = data.map(item => ({
                  value: item.count,
                  name: item.pct_range
              }));

              // 更新图表
              option.series[0].data = formattedData;
              myChart.setOption(option);
          },
          error: function (xhr, type, errorThrown) {
              console.error("获取股票涨跌幅数据失败:", errorThrown);
          }
      });
  }

  // 5. 定时刷新数据
  setInterval(update_data1, 1000 * 60 * 1); // 每分钟更新一次
  update_data1(); // 页面加载时立即获取数据
})();
//图2
(function() {
  var myColor = [
    "#006cff",
    "#60cda0",
    "#ed8884",
    "#ff9f7f",
    "#0096ff",
    "#9fe6b8",
    "#32c5e9",
    "#1d9dff"
  ];

  // 1. 实例化对象
  var myChart = echarts.init(document.querySelector(".bar2 .chart"));

  // 2. 配置项
  var option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
          type: 'shadow'
      },
      textStyle: {
        fontSize: 12,
      },
      formatter: '{b}:{c}'
	},
	toolbox: {
		show: false,
		orient: 'horizontal',
		feature: {
			dataZoom: {
				yAxisIndex: 'none'
			},
			dataView: {readOnly: false},
			magicType: {type: ['line', 'bar', 'stack', 'tiled']},
			restore: {},
			saveAsImage: {}
		}
	},
    xAxis: {
      show: true,
      type: 'category',
      axisLine: {
        show: false
      },
      axisLabel: {
        rotate: 0,  // 这里可以不用旋转
        color: "#ffffff",
        fontSize: "12"
      },
      barGap: '100%',
      barCategoryGap: '40%',
      data: ['Low', 'Medium', 'High'],  // 替换为成交量区间
    },
    yAxis: {
      type: "value",
      axisLine: {
        show: false
      },
      axisTick: {
        show: false
      },
      splitLine: {
        show: false,
      },
      axisLabel: {
        color: "#ffffff",
        fontSize: "12"
      },
    },
    series: [{
      data: [0, 0, 0],  // 初始化为空，后续用 AJAX 获取
      type: 'bar',
      label: {
          show: true,
          color: "#ffffff",
          position: 'top'
      },
      itemStyle: {
        color: function(params) {
          return myColor[params.dataIndex];
        }
      },
    }],
  };

  // 3. 渲染图表
  myChart.setOption(option);
  window.addEventListener("resize", function() {
    myChart.resize();
  });

  // 4. 更新数据
  function update_data2() {
      $.ajax({
          url: "/data?dname=data2",
          timeout: 10000, // 超时时间
          success: function (data) {

              let formattedData = [0, 0, 0];  // [Low, Medium, High]

              // 将返回的数据映射到正确的顺序
              data.forEach(item => {
                  if (item.vol_range === 'Low') {
                      formattedData[0] = item.count;
                  } else if (item.vol_range === 'Medium') {
                      formattedData[1] = item.count;
                  } else if (item.vol_range === 'High') {
                      formattedData[2] = item.count;
                  }
              });

              // 更新图表
              option.series[0].data = formattedData;
              myChart.setOption(option);
          },
          error: function (xhr, type, errorThrown) {
              console.error("获取最新成交量数据失败:", errorThrown);
          }
      });
  }

  // 5. 定时刷新数据
  setInterval(update_data2, 60000);
  update_data2();  // 页面加载时立即获取数据
})();

// 图3
(function() {
  var tableContainer = document.querySelector(".line .chart");

  // 创建表格
  var table = document.createElement("table");
  table.classList.add("stock-table");

  table.innerHTML = `
    <thead>
      <tr>
        <th>Date</th>
        <th>Stock Name</th>
        <th>Opening Price</th>
        <th>Closing Price</th>
      </tr>
    </thead>
    <tbody id="stock-data-body">
    </tbody>
  `;

  tableContainer.appendChild(table);

  // 轮播数据
  var stockData = [];
  var currentIndex = 0;
  var visibleRows = 5; // 只显示5条数据

  function updateStockTable() {
      $.ajax({
          url: "/data?dname=data3",
          timeout: 10000,
          success: function (data) {
              stockData = data;
              currentIndex = 0;
              renderTable();
          },
          error: function (xhr, type, errorThrown) {
              console.error("获取股票数据失败:", errorThrown);
          }
      });
  }

  function renderTable() {
      var tbody = document.getElementById("stock-data-body");
      tbody.innerHTML = ""; // 清空旧数据

      for (let i = 0; i < visibleRows; i++) {
          let index = (currentIndex + i) % stockData.length;
          let item = stockData[index];

          var row = document.createElement("tr");
          row.innerHTML = `
              <td>${item.trade_date}</td>
              <td>${item.ts_code}</td>
              <td>${item.open.toFixed(2)}</td>
              <td>${item.close.toFixed(2)}</td>
          `;
          tbody.appendChild(row);
      }
  }

  function scrollStockTable() {
      if (stockData.length > visibleRows) {
          var tbody = document.getElementById("stock-data-body");
          var rows = tbody.getElementsByTagName("tr");

          if (rows.length > 0) {
              // 给第一个元素加淡出动画
              rows[0].classList.add("fade-out");

              setTimeout(() => {
                  // 删除第一行
                  tbody.removeChild(rows[0]);

                  // 计算新的一行索引
                  let newIndex = (currentIndex + visibleRows) % stockData.length;
                  let newItem = stockData[newIndex];

                  // 创建新的一行并加上淡入动画
                  var newRow = document.createElement("tr");
                  newRow.classList.add("fade-in");
                  newRow.innerHTML = `
                      <td>${newItem.trade_date}</td>
                      <td>${newItem.ts_code}</td>
                      <td>${newItem.open.toFixed(2)}</td>
                      <td>${newItem.close.toFixed(2)}</td>
                  `;
                  tbody.appendChild(newRow);

                  // 更新索引
                  currentIndex = (currentIndex + 1) % stockData.length;
              }, 800); // 800ms 之后执行
          }
      }
  }

  updateStockTable();
  setInterval(scrollStockTable, 3000); // 每1秒滚动1条
  setInterval(updateStockTable, 1000 * 60 * 1); // 每分钟更新数据

})();


// 图4
(function () {
    var myChartBeta = echarts.init(document.querySelector("#beta-chart"));

    var optionBeta = {
        tooltip: {
            trigger: "item",
            formatter: function (params) {
                return `Stock Code: ${params.data[0]}<br>Beta 值: ${params.data[1].toFixed(2)}`;
            }
        },
        xAxis: {
            axisLabel: {
                color: "#ffffff", // 轴标签颜色
                fontSize: 12
            },
            axisLine: {
                lineStyle: { color: "#ffffff" } // 轴线颜色
            },
            name: "Stock Code",
            type: "category",
            data: []
        },
        yAxis: {
            axisLabel: {
                color: "#ffffff",
                fontSize: 12
            },
            axisLine: {
                lineStyle: { color: "#ffffff" }
            },
            name: "Coefficient",
            type: "value",
            min: function (value) { return value.min - 0.2; },
            max: function (value) { return value.max + 0.2; }
        },
        axisLabel: {
            color: "#ffffff",
            fontSize: "12"
          },
        series: [{
            name: "Beta Coefficient",
            type: "scatter",
            symbolSize: 12,
            data: [],
            itemStyle: {
                color: function (params) {
                    return params.data[1] > 1 ? "#ff4d4f" : "#4caf50";
                }
            }
        }]
    };

    myChartBeta.setOption(optionBeta);

    function updateBetaChart() {
        $.ajax({
            url: "/data?dname=data4",
            timeout: 10000,
            success: function (data) {
                if (!data || data.length === 0) {
                    console.error("Beta 数据为空");
                    return;
                }

                console.log("Beta 数据返回:", data);

                // 提取数据并格式化
                let stockCodes = data.map(item => item.ts_code);  // X 轴 Stock Code
                let betaValues = data.map(item => [item.ts_code, item.beta]);  // [Stock Code, Beta值]

                console.log("Stock Code:", stockCodes);
                console.log("Beta 散点数据:", betaValues);

                myChartBeta.setOption({
                    xAxis: {
                        data: stockCodes
                    },
                    yAxis: {
                        type: "value",
                        min: function (value) { return value.min - 0.2; }, // 自动缩放
                        max: function (value) { return value.max + 0.2; }
                    },
                    series: [{
                        type: "scatter",
                        name: "Beta Coefficient",
                        data: betaValues,
                        symbolSize: function (data) {
                            return Math.abs(data[1]) * 10 + 5; // 让散点大小更明显
                        },
                        itemStyle: {
                            color: function (params) {
                                return params.data[1] > 1 ? "#ff4d4f" : "#4caf50"; // 红=高风险，绿=低风险
                            }
                        }
                    }]
                });
            },
            error: function (xhr, type, errorThrown) {
                console.error("获取 Beta 数据失败:", xhr.status, errorThrown);
            }
        });
    }

    updateBetaChart();
    setInterval(updateBetaChart, 1000 * 60 * 1); // 每分钟更新
})();

function updateStockStatistics() {
    $.ajax({
        url: "/get_stock_statistics",
        type: "GET",
        dataType: "json",
        success: function (data) {
            $("#highest_close_stock").text(data.highest_close_stock + ": " + data.highest_close_price);
            $("#lowest_open_stock").text(data.lowest_open_stock + ": " + data.lowest_open_price);
            $("#avg_close_price").text(data.avg_close_price.toFixed(2));

            $("#highest_volume_stock").text(data.highest_volume_stock + ": " + data.highest_volume);
            $("#highest_pct_change_stock").text(data.highest_pct_change_stock + ": " + data.highest_pct_change + "%");
            $("#highest_amount_stock").text(data.highest_amount_stock + ": " + data.highest_amount);
        },
        error: function (err) {
            console.error("获取股票统计数据失败:", err);
        }
    });
}

// 每隔 1 分钟刷新一次数据
setInterval(updateStockStatistics, 60000);

// 页面加载时先获取一次数据
$(document).ready(function () {
    updateStockStatistics();
});


// **DCF 估值可视化**
// **全局定义 update_data6**
function update_data6() {
    let wacc = parseFloat(document.getElementById("wacc-input").value) / 100;
    let g = parseFloat(document.getElementById("g-input").value) / 100;
    let reinvestment_rate = parseFloat(document.getElementById("reinvestment-input").value) / 100;
    var myChart = echarts.init(document.querySelector(".pie2 .chart"));

    $.ajax({
      url: `/data?dname=data6&wacc=${wacc}&g=${g}&reinvestment_rate=${reinvestment_rate}`,
      timeout: 10000,
      success: function (data) {
        if (!data || data.length === 0) {
          console.error("DCF 数据为空");
          return;
        }

        let stocks = data.map(item => item.ts_code);
        let dcf_values = data.map(item => parseFloat(item.dcf_value.toFixed(2)));

        myChart.setOption({
          xAxis: { data: stocks },
          series: [
            { name: "DCF Value", data: dcf_values },
            { name: "DCF Trend", data: dcf_values }
          ]
        });
      },
      error: function (xhr, type, errorThrown) {
        console.error("获取 DCF 数据失败:", xhr.status, errorThrown);
      }
    });
}

// **DCF 估值可视化**
(function() {
  var myChart = echarts.init(document.querySelector(".pie2 .chart"));

  var option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: function(params) {
        let barData = params.find(p => p.seriesType === 'bar');
        if (barData) {
          return `${barData.name}<br>DCF Value: $${barData.value}B`;
        }
      },
    },
    legend: { bottom: "0%", textStyle: { color: "#ffffff", fontSize: "12" } },
    xAxis: {
      type: "category",
      axisLabel: { rotate: 45, color: "#ffffff", fontSize: "12" },
      data: [],
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#ffffff", fontSize: "12", formatter: '${value}B' }
    },
    series: [
      {
        name: "DCF Value",
        type: "bar",
        data: [],
        itemStyle: { color: "#006cff" },
        markPoint: {
          data: [
            { type: 'max', name: 'Highest', symbolSize: 50, label: { color: '#ffffff' } },
            { type: 'min', name: 'Lowest', symbolSize: 50, label: { color: '#ffffff' } }
          ]
        }
      },
      {
        name: "DCF Trend",
        type: "line",
        data: [],
        itemStyle: { color: "#ff9f7f" },
      }
    ]
  };

  myChart.setOption(option);

  // **初始加载**
  update_data6();
})();


(function () {
    var myChart = echarts.init(document.querySelector("#kline-chart"));

    var option = {
        title: {
            text: "Stock K-Line Chart",
            left: "center",
            textStyle: { color: "#fff" }
        },
        tooltip: { trigger: "axis" },
        dataZoom: [
            {
                type: "inside",
                start: 60,
                end: 100
            },
            {
                type: "slider",
                start: 60,
                end: 100,
                textStyle: {
                    color: "#ffffff"  // **修改滑块两侧标签的颜色**
                },
                handleStyle: {
                    color: "#7fffa3", // 滑块把手颜色
                    borderColor: "#ffffff"
                }
            }
        ],

        xAxis: {
            type: "category",
            data: [],
            axisLabel: { color: "#ffffff", fontSize: "12" }
        },
        yAxis: {
            scale: true,
            axisLabel: { color: "#ffffff", fontSize: "12" }
        },
        series: [{
            name: "K线图",
            type: "candlestick",
            data: [],
            itemStyle: {
                color: "#ff4d4f",  // 阳线（上涨，红色）
                color0: "#4caf50", // 阴线（下跌，绿色）
                borderColor: "#ff4d4f",
                borderColor0: "#4caf50"
            }
        }]
    };

    myChart.setOption(option);

    // 加载股票列表
    function loadStockList() {
        $.ajax({
            url: "/data?dname=stock_list",
            success: function (data) {
                console.log("股票列表:", data);
                var stockSelect = $("#stock-select");
                stockSelect.empty();

                if (data.length === 0) {
                    console.error("股票列表为空");
                    return;
                }

                data.forEach(stock => {
                    stockSelect.append(`<option value="${stock}">${stock}</option>`);
                });

                // 默认加载第一个股票的 K 线数据
                loadKlineData(data[0]);
            },
            error: function (xhr, type, errorThrown) {
                console.error("获取股票列表失败:", errorThrown);
            }
        });
    }

    // 加载指定股票的 K 线数据
    function loadKlineData(stockCode) {
        console.log("加载 K 线数据:", stockCode);
        $.ajax({
            url: "/data?dname=kline&stock=" + stockCode,
            success: function (data) {
                console.log("K 线数据返回:", data);

                if (!data.dates || !data.kline) {
                    console.error("K 线数据格式错误:", data);
                    return;
                }

                myChart.setOption({
                    xAxis: { data: data.dates },
                    series: [{ data: data.kline }],
                    dataZoom: [{ start: 60, end: 100 }] // 重新设定默认范围
                });
            },
            error: function (xhr, type, errorThrown) {
                console.error("获取 K 线数据失败:", errorThrown);
            }
        });
    }

    // 下拉框绑定事件
    $("#stock-select").on("change", function () {
        var selectedStock = $(this).val();
        console.log("选中股票:", selectedStock);
        loadKlineData(selectedStock);
    });

    // 页面加载时获取股票列表
    loadStockList();
})();

function loadStockList() {
    let stockSelectLstm = $("#stock-select-lstm");
    $.ajax({
        url: "/stock_list",
        success: function (data) {
            console.log("获取股票列表:", data);



            stockSelectKline.empty();
            stockSelectLstm.empty();

            if (data.length === 0) {
                console.error("股票列表为空");
                stockSelectLstm.append('<option value="">暂无股票</option>');
                return;
            }

            // 添加Stock Code到两个下拉框
            data.forEach(stock => {
                stockSelectLstm.append(`<option value="${stock}">${stock}</option>`);
            });

            // 默认选择第一只股票
            if (data.length > 0) {
                stockSelectLstm.val(data[0]);
                updatePrediction();  // 预测默认股票的未来价格
            }
        },
        error: function (xhr, type, errorThrown) {
            console.error("获取股票列表失败:", errorThrown);
        }
    });
}


// 图5 - LSTM 预测未来 7 天Closing Price
(function() {
  var myChartLSTM = echarts.init(document.querySelector("#lstm-chart"));  // 绑定 id="lstm-chart"

  var optionLSTM = {
    tooltip: { trigger: "axis" },
    legend: {
      bottom: "0%",
      textStyle: { color: "#ffffff", fontSize: "12" }
    },
    xAxis: {
      name: "Date",
      type: "category",
        axisLabel: {
            color: "#ffffff", // 轴标签颜色
            fontSize: 12
        },
        axisLine: {
            lineStyle: { color: "#ffffff" } // 轴线颜色
        },
      data: [],  // 动态填充
      axisLabel: { color: "#ffffff" }
    },
    yAxis: {
        splitLine: {
            show: false,
        },
        axisLabel: {
            color: "#ffffff", // 轴标签颜色
            fontSize: 12
        },
        axisLine: {
            lineStyle: { color: "#ffffff" } // 轴线颜色
        },
      name: "Closing Price",
      type: "value",
      axisLabel: { color: "#ffffff" }
    },
    series: [{
      name: "Predicted Closing Price",
      type: "line",
      data: [],  // 动态填充
      itemStyle: { color: "#ff9f7f" },
      label: { show: true, position: "top", color: "#ffffff" }
    }]
  };

  myChartLSTM.setOption(optionLSTM);

  // 获取 LSTM 预测数据并更新图表
  function updatePrediction() {
    var selectedStock = $("#stock-select-lstm").val();  // 获取选中的Stock Code

    $.ajax({
        url: "/data?dname=data5&stock=" + selectedStock,
        timeout: 10000,
        success: function (data) {
            if (!data || data.length === 0) {
                console.error("预测数据为空");
                return;
            }

            console.log("预测数据:", data);

            // 确保数据按 `id` 排序
            data.sort((a, b) => a.id - b.id);

            // 解析Date和Closing Price
            let dates = data.map(item => item.date);
            let prices = data.map(item => Number(item.predicted_close.toFixed(2)));

            // 更新 ECharts 配置
            myChartLSTM.setOption({
                xAxis: { data: dates },
                series: [{
                    name: "Closing Price",
                    type: "line",
                    data: prices
                }]
            });
        },
        error: function (xhr, type, errorThrown) {
            console.error("获取预测数据失败:", xhr.status, errorThrown);
        }
    });
  }

  // 监听窗口大小变化
  window.addEventListener("resize", function() {
    myChartLSTM.resize();
  });

  // 绑定下拉框事件（切换股票时更新预测数据）
  $("#stock-select-lstm").on("change", function () {
      updatePrediction();
  });

  // 加载股票列表
  function loadStockList() {
      $.ajax({
          url: "/stock_list",
          success: function (data) {
              console.log("获取股票列表:", data);

              var stockSelectLstm = $("#stock-select-lstm");
              stockSelectLstm.empty(); // 清空旧选项

              if (data.length === 0) {
                  console.error("股票列表为空");
                  stockSelectLstm.append('<option value="">暂无股票</option>');
                  return;
              }

              // 添加Stock Code到下拉框
              data.forEach(stock => {
                  stockSelectLstm.append(`<option value="${stock}">${stock}</option>`);
              });

              // 默认选择第一只股票，并自动更新预测数据
              if (data.length > 0) {
                  stockSelectLstm.val(data[0]);
                  updatePrediction();
              }
          },
          error: function (xhr, type, errorThrown) {
              console.error("获取股票列表失败:", errorThrown);
          }
      });
  }

  // 页面加载时获取股票列表
  $(document).ready(function () {
      loadStockList();
  });

})();
