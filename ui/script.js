'use strict'

const $ = document.querySelector.bind(document)

const $form = $('form')
const $input = $('#input')


$form.addEventListener('submit', handleSubmit)

function handleSubmit(event) {
    event.preventDefault()
    getPrediction()
}

function getPrediction() {
    const data = {
        text: $input.value
    }
    console.log(data)

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    }).then(response => response.json())
        .then(data => {
            console.log(data)
            $form.reset()
        })
        .catch((error) => {
            console.error('Error:', error)
        })
}

// Path: server/app.py
