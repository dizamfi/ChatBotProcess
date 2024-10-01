document.getElementById('upload-form').addEventListener('submit', function(e) {
  e.preventDefault();
  const formData = new FormData(this);
  fetch('/upload', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      document.getElementById('file-list').innerHTML = data.archivos.join('<br>');
  });
});

document.getElementById('question-form').addEventListener('submit', function(e) {
  e.preventDefault();
  const question = document.getElementById('question-input').value;
  if (question.trim() !== '') {
      const chatBox = document.getElementById('chat-box');
      const userQuestion = `<div class="user-question">${question}</div>`;
      chatBox.innerHTML += userQuestion;
      fetch('/ask', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: question })
      })
      .then(response => response.json())
      .then(data => {
          if (data.answer) {
              const answer = `<div class="bot-answer">${data.answer}</div>`;
              chatBox.innerHTML += answer;
          }
          document.getElementById('question-input').value = '';
      });
  }
});
