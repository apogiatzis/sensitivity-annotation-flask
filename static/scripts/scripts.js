$(function(){
	$('.annotate-btn').click(function(){
		var text = $('#annotate-txtarea').val();
		$.ajax({
            url: '/predict',
            dataType: "json",
            data: {'data':text},
			type: 'POST',
			success: function(response){
                console.log(response);
                $("#annotated-text-container").html(response.html)
			},
			error: function(error){
				console.log(error);
			}
		});
    });

    // Client Side Javascript to receive numbers.
    $(document).ready(function(){
        // start up the SocketIO connection to the server - the namespace 'test' is also included here if necessary
        var socket = io.connect('http://' + document.domain + ':' + location.port + '/socket');
        // this is a callback that triggers when the "my response" event is emitted by the server.
        socket.on('annotations', function(msg) {
            $("#annotated-text-container").html(msg.data);
            $('#spinner').css('display', 'none');
            clearTimeout(spinnerTimer);
        });

        //setup before functions
        var typingTimer;                //timer identifier
        var doneTypingInterval = 1500;  //time in ms, 5 second for example
        var $input = $('#annotate-txtarea');
        var spinnerTimer;                //Spinner timer identifier
        var spinnerTimeOutInterval = 20000;  //time in ms, 20 seconds for timeout


        //on keyup, start the countdown
        $input.on('keyup', function () {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(doneTyping, doneTypingInterval);
        });
        
        //on keydown, clear the countdown 
        $input.on('keydown', function () {
            clearTimeout(typingTimer);
        });

        function removeSpinner() {
            $('#spinner').css('display', 'none');
            $('span.timeout-msg').css('display','inline-block');
        }
        
        //user is "finished typing," do something
        function doneTyping () {
            $('#spinner').css('display', 'inline-block');
            $('span.timeout-msg').css('display','none');
            var text = $('#annotate-txtarea').val();
            socket.emit('predict', {data: text});
            spinnerTimer = setTimeout(removeSpinner, spinnerTimeOutInterval);
        }
    });
    
});
