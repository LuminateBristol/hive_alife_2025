$(document).ready(function() {

    redisData = {}
    metaData = {}
    metricData = {}
    runtime = $('input[name=sim_runtime]').val()
    scenario = $('input[name=scenario]').val()
    
    $('#btns #reload').prop('disabled', true)
    $('#btns #play').prop('disabled', true)
    $('#btns .step').prop('disabled', true)
    $('#btns #pause').prop('disabled', true)

    function fetchRedis() {
        el = $('#fetchRedis')
        el.prop('disabled', true)
        $('#runEx').prop('disabled', true)
        $.ajax({
            url: '/fetch-redis/' + runtime + '/' + scenario,
            method: 'GET',
            success: function (data) {
                console.log("Success: fetch redis data")
                redisData = flattenData(data['simdata'])
                metaData = unpackData(data['metadata'])
                metricData = data['metricdata']
                populateMetrics()
            },
            error: function (error) {
                console.log("Error: fetch redis data")
            },
            complete: function () {
                el.prop('disabled', false)
                $('#runEx').prop('disabled', false)
                $('#btns #reload').prop('disabled', false)
                $('#btns #play').prop('disabled', false)
                $('#btns .step').prop('disabled', false)
                $('#btns #pause').prop('disabled', false)
            }
        });    
    }
    
    // Neccessary to flatten object
    function flattenData(data) {
        newData = {}
        for (const [key, value] of Object.entries(data)) {
            newData[key] = value[key]            
        }   
        return newData
    }

    function unpackData(data) {
        newData = {}
        for (const [key, value] of Object.entries(data)) {
            newData[key] = JSON.parse(value)
        }   
        return newData
    }

    $('#fetchRedis').on('click', fetchRedis)

    var canvas = document.querySelector("#arena")   // Get access to HTML canvas element
    var ctx = canvas.getContext("2d")
    var canvasWidth = canvas.width = $('input[name=arena_w]').val()
    var canvasHeight = canvas.height = $('input[name=arena_h]').val()

    canvasData = {
        'robot_r': $('input[name=robot_radius]').val(),
        'box_r': $('input[name=box_radius]').val(),
        'deposit_zones': $('input[name=deposit_zones]').val(),
        'timestep': 1
    }
    
    function drawDepositZones() {
        ctx.beginPath()
        ctx.setLineDash([5, 5])
        ctx.strokeStyle = '#4e733b'
        ctx.moveTo(500-25, 0);
        ctx.lineTo(500-25, 500);
        ctx.stroke()
        ctx.closePath()
    }

    function drawRobot(id, x, y, is_faulty=false) {
        r = Number(canvasData['robot_r'])
        ctx.beginPath()
        // ctx.fillStyle = '#6f777d'
        ctx.setLineDash([])
        ctx.lineWidth = 2;
        if (is_faulty) {
            ctx.strokeStyle = 'red';
        }
        else {
            ctx.strokeStyle = '#1f1f1f';
        }        
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.stroke()
        // ctx.fill()
        if (x > canvasWidth-r-19) {
            t0 = x-r-10
        } else {
            t0 = x+r+1
        }
        if (y > canvasHeight-r-10) {
            t1 = y-r-4
        } else {
            t1 = y+r
        }
        
        ctx.font = "10px Arial";
        ctx.fillStyle = '#1f1f1f';
        ctx.fillText("r"+id, t0, t1);
        ctx.closePath()
    }

    function drawCamSensor(x, y, r) {
        ctx.beginPath()
        // ctx.fillStyle = 'red'
        // ctx.setLineDash([5, 5])
        ctx.strokeStyle = '#f2f2f2'
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.stroke()
        ctx.closePath()
        // ctx.fill()
    }

    function drawBox(x, y) {
        r = canvasData['box_r'] * 1.3
        ctx.beginPath()
        ctx.fillStyle = 'blue'
        c1 = x - r/2
        c2 = y - r/2
        ctx.fillRect(c1,c2,r,r);
        ctx.fill()
        ctx.closePath()
    }
    // Function clears the canvas
    function clearCanvas() {
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    }

    function drawFrame() {
        clearCanvas()
        drawDepositZones()
        ts = canvasData['timestep']
        d = redisData[ts]
        rob_c = JSON.parse(d['robot_coords'])
        box_c = JSON.parse(d['box_coords'])
        cam_r = JSON.parse(d['camera_range'])
        faults = metaData['fault_count'][0]
        no_box = d['no_boxes']
        no_rob = d['no_robots']
        
        i = 0
        for (const c of rob_c) {
            r = cam_r[i]
            drawCamSensor(c[0], c[1], r)
            i++
        }

        i = 1
        for (const c of rob_c) {
            if (i <= faults) {
                is_faulty = true
            }
            else {
                is_faulty = false
            }
            drawRobot(i, c[0], c[1], is_faulty)
            i++
        }

        for (const c of box_c) {
            drawBox(c[0], c[1])
        }
    }

    $('#btns #reload').on('click', function(){
        canvasData['timestep'] = 1
        drawFrame()
    });

    $('#btns .step').on('click', function(){
        step = Number($(this).data('step'))
        canvasData['timestep'] += step
        if (canvasData['timestep'] >= Number(runtime)) {
            canvasData['timestep'] = Number(runtime)
        }
        if (canvasData['timestep'] < 0) {
            canvasData['timestep'] = 0
        }
        drawFrame()
        populateMetrics()
    });

    paused = false
    $('#btns #pause').on('click', function(){
        paused = true
    });

    running = false
    $('#btns #play').on('click', function(){
        if (running) {
            if (paused) {
                paused = false
            }
            return
        }
        interval = window.setInterval(function(){
            running = true
            if (paused) {
                return
            }
            canvasData['timestep'] += 1
            if (canvasData['timestep'] >= Number(runtime)) {
                clearInterval(interval)
            }
            drawFrame()
            populateMetrics()
        }, 100);        
    })

    // Displays metric data for current timestep and selected metric
    function populateMetrics() {
        t = canvasData['timestep']
        data = metricData[t]
        metric = metaData['metrics']
        ag_id = Number($('.metrics select').val())
        showMetricData(data, ag_id)
    }

    function showMetricData(data, ag_id) {
        html = ''
        for (const [metric, arr] of Object.entries(data)) {
            data_ = JSON.parse(arr.replaceAll('NaN', 'null'))
            value = data_[ag_id]
            html += '<p>'+metric+': <span class="val">'+value+'</span></p>'
        }

        $('#metricData').html(html)
    }
})

