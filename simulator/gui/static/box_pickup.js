$(document).ready(function() {
    
    $('#runEx').click(function() {
        ex_id = $('#box_pickup_controls select[name=cfg]').val()
        faults = $('#box_pickup_controls input[name=no_faults]').val()

        el = $('#runEx')
        el.prop('disabled', true)
        $('#fetchRedis').prop('disabled', true)
        $.ajax({
            url: '/run-ex/' + ex_id + '/' + faults,
            method: 'POST',
            success: function (data) {
                console.log("Run sim successful")
                // random_seed = data
            },
            error: function (error) {
                console.log("Error: ")
            },
            complete: function () {
                el.prop('disabled', false)
                $('#fetchRedis').prop('disabled', false)
            }
        })
    })

});