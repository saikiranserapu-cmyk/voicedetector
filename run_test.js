const fs = require('fs');

async function test() {
  try {
    const fd = new FormData();
    fd.append('audio', new Blob(['1234']), 'test.webm');
    const res = await fetch('http://127.0.0.1:3000/api/predict', { 
      method: 'POST', 
      body: fd 
    });
    const data = await res.json();
    console.log('STATUS:', res.status, data);
  } catch (err) {
    console.error('FAIL:', err);
  }
}

test();
