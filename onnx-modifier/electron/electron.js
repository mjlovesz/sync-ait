
// electron 模块可以用来控制应用的生命周期和创建原生浏览窗口
const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require("node:child_process")
const { EventEmitter } = require('events')
var fs = require('fs')
const path = require('path')

const createWindow = () => {
    // 创建浏览窗口
    const mainWindow = new BrowserWindow({
        width: 1000,
        height: 900,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js')
        }
    })

    // 加载 index.html
    mainWindow.loadFile('static/electron.html')
}

// 这段程序将会在 Electron 结束初始化
// 和创建浏览器窗口的时候调用
// 部分 API 在 ready 事件触发后才能使用。
app.whenReady().then(() => {
    createWindow()

    app.on('activate', () => {
        // 在 macOS 系统内, 如果没有已开启的应用窗口
        // 点击托盘图标时通常会重新创建一个新窗口
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })

    let handle_msg = new ElectronMsgHandelManager()
    ipcMain.handle('message', (event, path, msg_send) => {
        return handle_msg.handleMessage(event, path, msg_send)
    })
})

// 除了 macOS 外，当所有窗口都被关闭的时候退出程序。 因此, 通常
// 对应用程序和它们的菜单栏来说应该时刻保持激活状态, 
// 直到用户使用 Cmd + Q 明确退出
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})

// 在当前文件中你可以引入所有的主进程代码
// 也可以拆分成几个文件，然后用 require 导入。


class PythonIPC {
    constructor() {
        let app_path = path.join(__dirname, "..", "app.py")
        this.process = spawn('python', [app_path])
        this.msg_event = new EventEmitter()

        this.process.stdout.on("data", (data_text) => {
            data_text = data_text.toString()
            console.debug(`${data_text}`)
            let data_array = data_text.split(/\r?\n/)
            if (data_array.length < 6) {
                return
            }
            data_array = data_array.slice(-6)

            if (data_array[0] == "" && data_array[1] == "" && data_array[2] == ">>" && data_array[4] == "" && data_array[5] == "") {
                let data = data_array[3]
                this.msg_event.emit('data', data)
            }
        })
    }

    send(path, msg_send) {
        console.log(this.process.pid)
        return new Promise((resolve) => {
            this.msg_event.once("data", (data) => {
                console.debug("recv", `${data}`)
                let { msg, status, file } = JSON.parse(data)
                if (file) {
                    file = fs.readFileSync(file)
                }
                resolve([status, msg, file])
            })
            let send_obj_data = { path, msg: msg_send }
            let send_obj_str = JSON.stringify(send_obj_data, null, 1)

            console.debug("send", `\n${send_obj_str}\n`)
            this.process.stdin.write(`${send_obj_str}\n`)
            this.process.stdin.write(`\n\n`)
        })
    }
}

class ElectronMsgHandelManager {
    constructor() {
        this.map_ipc = new Map()
    }

    handleMessage(event, path, msg_send) {
        let senderId = event.processId
        if (!this.map_ipc.has(senderId)) {
            this.map_ipc.set(senderId, new PythonIPC())
        }

        let python_ipc = this.map_ipc.get(senderId)
        return python_ipc.send(path, msg_send)
    }
}