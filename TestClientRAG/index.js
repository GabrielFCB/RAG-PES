const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");

const app = express();
const PORT = 3000;

// Middleware para servir arquivos estáticos e processar JSON
app.use(express.static("public"));
app.use(bodyParser.json());

app.post("/invoke-openai", async (req, res) => {
  const { question } = req.body;

  try {
    const response = await axios.post(
      "http://server:8000/rag-conversation/invoke",
      {
        input: {
          chat_history: [["", ""]],
          question: question,
        },
        config: {},
        kwargs: {},
      }
    );

    res.json({ success: true, data: { answer: response.data.output } }); // Altere para response.data.output
    console.log(response.data.output); // Log para verificar a resposta
  } catch (error) {
    console.error(
      `Error: ${error.response.status} - ${error.response.data.detail}`
    );
    res.json({ success: false, error: error.response.data.detail });
  }
});

app.listen(PORT, async () => {
  console.log(`Servidor rodando na porta ${PORT}`);
  const open = await import("open");
  open.default(`http://localhost:${PORT}`);
});
