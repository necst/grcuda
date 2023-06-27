const fs = require("fs")

/**
 * Reads the MTX file
 * @param path path of the input file
 * @param normalize weather or not to normalize val
 * @returns {{val: *[], x: *[], y: *[]} the coo matrix
 */
const readDataset = (path, normalize) => {
    const file = fs.readFileSync(path)
    const allLines = file.toString().split("\n")
    const neededLines = allLines.filter(line => !line.includes("%"))
    const header = allLines.filter(line => line.includes("%"))
    let x = []
    let y = []
    let val = []
    neededLines.forEach((line, i) => {
        let [v_x, v_y, v_val] = line.split(" ").map(v => parseFloat(v))
        if (v_y == undefined) return
        if (v_val == undefined) val.push(1)
        else val.push(v_val)
        x.push(v_x)
        y.push(v_y - 1) // WHY?
    })

    x.shift()
    y.shift()
    val.shift()
    const norm = Math.sqrt(val.reduce((acc, cur) => acc + cur * cur))
    if(normalize) val = val.map(value => value / norm)
    return {x, y, val}

}

module.exports = {
    readDataset
}