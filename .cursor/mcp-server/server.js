const express = require('express');
const app = express();
const port = 3000;

// In-memory storage for thoughts
const thoughts = [];

app.use(express.json());

// CORS middleware
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Route to handle the think tool
app.post('/think', (req, res) => {
  const { thought } = req.body.parameters;
  
  if (!thought) {
    return res.status(400).json({ error: 'Thought parameter is required' });
  }
  
  const timestamp = new Date().toISOString();
  thoughts.push({ thought, timestamp });
  
  return res.json({
    result: {
      message: "Thought recorded successfully.",
      timestamp,
      thoughtCount: thoughts.length
    }
  });
});

// Get all thoughts (for debugging)
app.get('/thoughts', (req, res) => {
  res.json(thoughts);
});

// Start the server
app.listen(port, () => {
  console.log(`MCP Think Tool server listening at http://localhost:${port}`);
}); 