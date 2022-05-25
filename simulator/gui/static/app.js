$(document).ready(function() {

    redisData = {}
    metaData = {}
    metricData = {}
    ad_pred = {}
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
                adPred = unpackData(data['ad_pred'])
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
            if (value == null) {
                newData[key] = value    
            } else {
                newData[key] = JSON.parse(value)
            }
        }   
        return newData
    }

    $('#fetchRedis').on('click', fetchRedis)

    function drawAxes() {
         // Add axis labels to arena
        yaxis = $('#yaxis')
        xaxis = $('#xaxis')
        cpos = $('#arena').offset()
        yaxis.offset({'left': cpos.left-yaxis.width(), 'top': cpos.top-5})
        xaxis.offset({'left': cpos.left-5, 'top': cpos.top+504})

        var ycanv = document.querySelector("#yaxis")
        var yctx = ycanv.getContext("2d")
        yh = ycanv.height = 513
        yw = ycanv.width = 100
        t0 = yw-10
        t1 = yw
        text_pos = t0-30
        ticks = [6,106,206,306,406,508]
        text = [500,400,300,200,100]
        $.each(ticks, function( index, tick ){
            yctx.beginPath()
            yctx.lineWidth = 2
            yctx.strokeStyle = '#2b2b2b'
            yctx.moveTo(t0, tick)
            yctx.lineTo(t1,tick)
            yctx.stroke()
            if (index < 5) {
                yctx.font = "14px Helvetica bold";
                yctx.fillStyle = '#2b2b2b';
                yctx.fillText(text[index], text_pos, tick+5);
            }
            yctx.closePath()
        })
        yctx.fillText(0, text_pos+15, 512);

        var xcanv = document.querySelector("#xaxis")
        var xctx = xcanv.getContext("2d")
        xh = xcanv.height = 100
        xw = xcanv.width = 520
        t0 = 0
        t1 = 10
        text_pos = t0+26
        ticks = [6,106,206,306,406,508]
        text = [0,100,200,300,400,500]
        $.each(ticks, function( index, tick ){
            xctx.beginPath()
            xctx.lineWidth = 2
            xctx.strokeStyle = '#2b2b2b'
            xctx.moveTo(tick, t0)
            xctx.lineTo(tick, t1)
            xctx.stroke()
            if (index > 0) {
                xctx.font = "14px Helvetica bold";
                xctx.fillStyle = '#2b2b2b';
                xctx.fillText(text[index], tick-14, text_pos);
            }
            xctx.closePath()
        })    
        xctx.fillText(0, 2, text_pos);

        $('#height_lab').offset({
            'top': yaxis.offset().top+180, 'left': yaxis.offset().left+20})
        $('#width_lab').offset({
            'top': xaxis.offset().top+50, 'left': xaxis.offset().left+180})
    }

    // Main canvas drawing

    var canvas = document.querySelector("#arena")   // Get access to HTML canvas element
    var ctx = canvas.getContext("2d")
    var canvasWidth = canvas.width = $('input[name=arena_w]').val()
    var canvasHeight = canvas.height = $('input[name=arena_h]').val()

    drawAxes()

    canvasData = {
        'robot_r': $('input[name=robot_radius]').val(),
        'box_r': $('input[name=box_radius]').val(),
        'deposit_zones': $('input[name=deposit_zones]').val(),
        'timestep': 1
    }
    
    function drawDepositZones() {
        ctx.beginPath()
        ctx.setLineDash([5, 5])
        ctx.lineWidth = 2
        ctx.strokeStyle = '#549154'
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
        ctx.lineWidth = 2
        ctx.strokeStyle = '#1f1f1f'
        if (is_faulty) {
            drawPentagon(x,y,r,ctx)
        }
        else {
            ctx.arc(x, y, r, 0, 2 * Math.PI);
        }        
        
        ctx.stroke()

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
        
        // Draw robot ID
        // ctx.font = "10px Arial";
        // ctx.fillStyle = '#1f1f1f';
        // ctx.fillText("r"+id, t0, t1);

        drawHeading(i-1,x,y,r)

        ctx.closePath()
    }

    function drawPentagon(x,y,r) {
        step  = 2 * Math.PI / 5,//Precalculate step value
        shift = (Math.PI / 180.0) * -18;//Quick fix ;)

        //cxt.moveTo (Xcenter +  size * Math.cos(0), Ycenter +  size *  Math.sin(0));          

        for (var i = 0; i <= 5;i++) {
            var curStep = i * step + shift;
            ctx.lineTo (x + r * Math.cos(curStep), y + r * Math.sin(curStep));
        }
    }

    function drawHeading(rid,x,y,r) {
        ts = canvasData['timestep']
        heading = JSON.parse(metricData[ts]['heading'])[rid]
        unit_y = Math.sin(heading) // -sin(-heading)
        unit_x = -Math.cos(heading) // -cos(-heading)
        start = [x+(r+0.1)*unit_x, y+(r+0.1)*unit_y]
        length = r*1.7
        end = [x+length*unit_x, y+length*unit_y]
        ctx.beginPath()
        ctx.lineWidth = 2
        ctx.strokeStyle = '#65bf65'
        ctx.moveTo(start[0], start[1])
        ctx.lineTo(end[0], end[1])
        ctx.stroke()
        ctx.closePath()
    }

    function drawCamSensor(x, y, r) {
        ctx.beginPath()
        // ctx.fillStyle = 'red'
        ctx.setLineDash([])
        ctx.lineWidth = 1
        ctx.strokeStyle = '#d1d1d1'
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.stroke()
        ctx.closePath()
        // ctx.fill()
    }

    function drawFault(x, y) {
        r = Number(canvasData['robot_r'])*1.1
        ctx.beginPath()
        // ctx.fillStyle = '#6f777d'
        ctx.setLineDash([])
        ctx.lineWidth = 2
        ctx.strokeStyle = 'red';
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.stroke()
        ctx.closePath()
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
        ap = adPred[ts]
        rob_c = JSON.parse(d['robot_coords'])
        box_c = JSON.parse(d['box_coords'])
        cam_r = JSON.parse(d['camera_range'])
        faults = metaData['fault_count'][0]
        no_box = d['no_boxes']
        no_rob = d['no_robots']
        
        $('#time_txt').html(Number(ts/50-0.02).toFixed(2))

        i = 0
        for (const c of rob_c) {
            r = cam_r[i]
            drawCamSensor(c[0], 500-c[1], r)
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
            drawRobot(i, c[0], 500-c[1], is_faulty)
            if (ap != null && ap[i-1]) {
                drawFault(c[0], 500-c[1])
            }
            i++
        }

        for (const c of box_c) {
            drawBox(c[0], 500-c[1])
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
        }, 10);        
    })

    // Displays metric data for current timestep and selected metric
    function populateMetrics() {
        t = canvasData['timestep']
        data = metricData[t]
        // metric = metaData['metrics']
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

