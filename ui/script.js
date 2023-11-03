// Path: ui/script.js
'use strict'

const $ = document.querySelector.bind(document)

const $form = $('form')
const $input = $('#input')
const $body = $('#body')
const $input_card = $('#input-card')


$form.addEventListener('submit', handleSubmit)

function handleSubmit(event) {
    event.preventDefault()
    getPrediction()
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
            };

        })
        .catch((error) => {
            console.error('Error:', error)
        })
}

function changeBackground() {
    $body.className = 'body_spam'
    $input_card.classList.add('input-main-spam')

}

// Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
