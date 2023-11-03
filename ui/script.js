'use strict'

const $ = document.querySelector.bind(document)

const $form = $('form')
const $input = $('#input')
const $body = $('#body')


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
            const { prediction } = data;
            if prediction[0][0] == 'spam' {changeBackground()});
            $form.reset() 
        })
        .catch((error) => {
            console.error('Error:', error)
        })
}

function changeBackground() {
    $body.classList.toggle = ('body_spam')
}

// Path: server/app.py
