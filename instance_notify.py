import asyncio
import aiohttp
import sys
from urllib.parse import urljoin
import json


_retrieved_metadata_bot_ip = None
discord_bot_ip = '10.142.0.3'
metadata_server = '169.254.169.254'
ip_url = ''

async def get_metadata(metadata_path):
    headers = {
        'Metadata-Flavor': 'Google'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(urljoin('http://'+metadata_server, metadata_path), headers=headers) as resp:
            if resp.status < 200 or resp.status > 299:
                return None
                
            return await resp.text()

async def get_external_ip():
    return await get_metadata('/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip')

async def get_instance_name():
    return await get_metadata('/computeMetadata/v1/instance/name')
    
async def get_discord_bot_ip():
    global _retrieved_metadata_bot_ip
    
    if _retrieved_metadata_bot_ip is not None:
        return _retrieved_metadata_bot_ip
    
    metadata_ip = await get_metadata('http://metadata/computeMetadata/v1/project/discordBotIP')
    
    if metadata_ip is None:
        return discord_bot_ip
    else:
        _retrieved_metadata_bot_ip = metadata_ip
        return metadata_ip
    
async def notify_discord(msg):
    _bot_ip = await get_discord_bot_ip()
    
    _, notify_writer = await asyncio.open_connection(_bot_ip, 8005)
    m = json.dumps({'type': 'notify', 'level': 'log', 'message': msg})
    notify_writer.write(m.encode('utf-8'))
    notify_writer.close()
    
async def main():
    mode = sys.argv[1]
    
    instance_name = await get_instance_name()
    ext_ip = await get_external_ip()
    discord_bot_ip = await get_discord_bot_ip()
    
    if mode == 'ident':
        print("name: {}".format(instance_name))
        print("extIP: {}".format(ext_ip))
        print("discordBotIP: {}".format(discord_bot_ip))
    elif mode == 'ping':
        await notify_discord("PING from `{}` (external IP `{}`).".format(instance_name, ext_ip))
    elif mode == 'startup':
        await notify_discord("Instance `{}` started up with external IP `{}`.".format(instance_name, ext_ip))
    elif mode == 'shutdown':
        await notify_discord("Instance `{}` shutting down.".format(instance_name))
        
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())