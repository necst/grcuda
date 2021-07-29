const ws = new WebSocket("ws://localhost:8080")
const sendWSMessage = document.getElementById("btn-send-msg-ws")

ws.addEventListener("open", (evt) => {
  console.log("Connection to websocket is open at ws://localhost:8080")
})

ws.addEventListener("message", (evt) => {
  console.log(`Received: ${evt.data}`)
})

sendWSMessage.onclick = () => {

  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Beginning computation on ${computationType}`)
  ws.send(computationType)
}

console.log("ciaone")