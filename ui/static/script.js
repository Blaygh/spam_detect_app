// Path: ui/script.js
'use strict'

const $ = document.querySelector.bind(document)

const $form = $('form')
const $input = $('#input')
const $body = $('#body')
const $input_card = $('#input-card')
const $popup = $('#mypop-up')



window.addEventListener("load", test);

function test(){
    fetch('http://127.0.0.1:5000/handshake', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(response => response.json())
        .then(data => {
            const{test} = data;
            console.log(test);
            if (test == 1) {
                console.log('App and running')
            }else{
                serverDown()
                console.log('server down')
            }
        }).catch(error=>{
            console.error('Error', error);
            serverDown()

        });
};

function serverDown(){
    $popup.style.display = 'block'; 
    $input_card.style.display = 'none';
    console.log('server down');
}

$form.addEventListener('submit', handleSubmit)

function handleSubmit(event) {
    event.preventDefault()
    if (!value_check()) return
    getPrediction()
}

function value_check() {
    if ($input.value == "") {
        alert("Please enter some text in the input field");
        return false;
    }
    return true;
}


function getPrediction() {
    const data = {
        text: $input.value
    }
    // console.log(data)

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    }).then(response => response.json())
        .then(data => {
            $form.reset() 

            const { prediction } = data;
            console.log(prediction[0][0])
            if (prediction[0][0] == 'spam') 
            {
                // call changeBackground()
                changeBackground()
            }else{
                changeBackground_ham()
            };

        })
        .catch((error) => {
            console.error('Error:', error)
        })
}

async function changeBackground() {
    $body.className = 'body_spam'
    $input_card.classList.add('input-main-spam')

    //for the blinkning effect on the input card
    for (let i = 0; i < 6; i++) {
        await new Promise(resolve => setTimeout(resolve, 300)); // Wait for 1/3 second
        $input_card.classList.toggle('input-main-spam-shadow')

    }

    await new Promise(resolve => setTimeout(resolve, 1500)); 
    $body.className = 'body_norm'
    $input_card.classList.remove('input-main-spam')

}

// // changing of background for ham

async function changeBackground_ham() {

    //for the blinkning effect on the input card
    for (let i = 0; i < 4; i++) {
        await new Promise(resolve => setTimeout(resolve, 300)); // Wait for 1/3 second
        $input_card.classList.toggle('input-main-ham-shadow')

    }

}




// changing of background
// setInterval(bg_rotate, 10000);

// function bg_rotate(){
//     var colors = ['background_2', 'background_3', 'background_4','background_5'];
//     var random_background = colors[Math.floor(Math.random() * colors.length)];

//     $body.classList.toggle(random_background);
// }

// Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
