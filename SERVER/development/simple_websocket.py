
import asyncio

import websockets

PORT = 8080

async def handler(websocket):
    async for msg in websocket:
        print(msg)


async def main():
    print(PORT)
    async with websockets.serve(handler, port=PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
