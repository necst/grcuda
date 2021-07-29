import express from 'express'
import WebSocket from 'ws'
import http from 'http'
import { GrCUDAProxy } from './GrCUDAProxy'

const app = express()
const server = http.createServer(app)
const PORT = 8080

const wss = new WebSocket.Server({ server })

wss.on('connection', (ws: WebSocket) => {
  console.log("A new client connected")
  const grCUDAProxy = new GrCUDAProxy(ws)

  ws.on('message', async (message: string) => {
    await grCUDAProxy.beginComputation(message)
  })

})

app.get('/', (req: any, res: any) => {
  res.send("Everithing is working properly")
})

server.listen(PORT, () => console.log(`Running on port ${PORT}`))