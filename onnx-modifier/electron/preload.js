// 所有的 Node.js API接口 都可以在 preload 进程中被调用.
// 它拥有与Chrome扩展一样的沙盒。

const { ipcRenderer, contextBridge } = require('electron')


function fetch_electron(path, msg) {
    console.log(typeof msg)
    console.log(msg)
    return ipcRenderer.invoke('message', path, msg).then((result) => {
        let [status, msg, file] = result
        return [status, msg, file]
    })
}


window.addEventListener('DOMContentLoaded', () => {
    contextBridge.exposeInMainWorld('is_electron', true)
    contextBridge.exposeInMainWorld('fetch_electron', fetch_electron)
  })