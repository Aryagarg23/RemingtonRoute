// Minimal Zip puzzle demo
// Rules enforced by this demo:
// - Grid is orthogonal (rows x cols).
// - Click or drag to visit adjacent cells; cells cannot be revisited.
// - Numbered cells must be visited in ascending order when encountered.
// - Final check asserts path covers all cells and numbers were visited in order.

(function(){
  const level = {
    rows: 6,
    cols: 6,
    // numbered cells: r, c, num (0-indexed r/c)
    numbers: [
      {r:0,c:1,num:1},
      {r:2,c:2,num:2},
      {r:3,c:4,num:3},
      {r:5,c:5,num:4}
    ]
  }

  function buildGrid(container, lvl){
    const grid = document.createElement('div')
    grid.className = 'zip-grid'
    grid.style.gridTemplateColumns = `repeat(${lvl.cols}, 1fr)`
    grid.style.gridTemplateRows = `repeat(${lvl.rows}, 1fr)`

    const cellMap = []
    for(let r=0;r<lvl.rows;r++){
      cellMap[r] = []
      for(let c=0;c<lvl.cols;c++){
        const wrapper = document.createElement('div')
        wrapper.className = 'zip-cell-wrapper'
        const cell = document.createElement('div')
        cell.className = 'zip-cell'
        cell.dataset.r = r
        cell.dataset.c = c
        // find number
        const found = lvl.numbers.find(n=>n.r===r && n.c===c)
        if(found){
          cell.classList.add('number')
          cell.textContent = found.num
          cell.dataset.number = found.num
        }
        wrapper.appendChild(cell)
        grid.appendChild(wrapper)
        cellMap[r][c] = {el:cell, visited:false}
      }
    }
    return {grid, cellMap}
  }

  function neighbors(r,c,rows,cols){
    return [{r:r-1,c},{r:r+1,c},{r,c:c-1},{r,c:c+1}].filter(p=>p.r>=0 && p.r<rows && p.c>=0 && p.c<cols)
  }

  function init(){
    const container = document.getElementById('zip-game')
    if(!container) return
    container.innerHTML = ''
    const {grid, cellMap} = buildGrid(container, level)
    container.appendChild(grid)

    const state = {path:[], expectedNextNumber:1, cellsTotal: level.rows*level.cols, dragging:false}

    function setMsg(text,ok){
      const el = document.getElementById('zip-msg')
      if(!el) return
      el.textContent = text
      el.style.color = ok? 'green' : 'var(--muted)'
    }

    function visitCell(r,c){
      const cell = cellMap[r][c]
      if(cell.visited) return false
      // adjacency check
      if(state.path.length>0){
        const last = state.path[state.path.length-1]
        const dr = Math.abs(last.r - r), dc = Math.abs(last.c - c)
        if(!((dr===1 && dc===0) || (dr===0 && dc===1))) return false
      }

      // number order check: if cell has a number, it must be the next expected number or higher? enforce equality
      const num = cell.el.dataset.number ? Number(cell.el.dataset.number) : null
      if(num !== null){
        if(num !== state.expectedNextNumber){
          setMsg(`You hit number ${num} but expected ${state.expectedNextNumber}`, false)
          return false
        }
        state.expectedNextNumber++
      }

      cell.visited = true
      cell.el.classList.add('visited')
      const seq = state.path.length+1
      cell.el.dataset.seq = seq
      state.path.push({r,c})
      return true
    }

    function reset(){
      state.path = []
      state.expectedNextNumber = 1
      for(let r=0;r<level.rows;r++) for(let c=0;c<level.cols;c++){
        const cell = cellMap[r][c]
        cell.visited = false
        cell.el.classList.remove('visited')
        delete cell.el.dataset.seq
      }
      setMsg('')
    }

    // mouse / touch handlers
    grid.addEventListener('pointerdown', e=>{
      grid.setPointerCapture(e.pointerId)
      state.dragging = true
      const cell = e.target.closest('.zip-cell')
      if(!cell) return
      const r = Number(cell.dataset.r), c = Number(cell.dataset.c)
      visitCell(r,c)
    })
    grid.addEventListener('pointermove', e=>{
      if(!state.dragging) return
      const cell = e.target.closest('.zip-cell')
      if(!cell) return
      const r = Number(cell.dataset.r), c = Number(cell.dataset.c)
      visitCell(r,c)
    })
    window.addEventListener('pointerup', e=>{ state.dragging=false })

    document.getElementById('zip-reset').addEventListener('click', ()=>{ reset() })
    document.getElementById('zip-check').addEventListener('click', ()=>{
      // validate
      if(state.path.length !== state.cellsTotal){
        setMsg(`Path covers ${state.path.length}/${state.cellsTotal} cells — fill every cell`, false)
        return
      }
      // ensure all numbers were visited in order
      const maxNumber = Math.max(...level.numbers.map(n=>n.num))
      if(state.expectedNextNumber-1 !== maxNumber){
        setMsg(`You did not visit all numbered cells in order (stopped at ${state.expectedNextNumber-1})`, false)
        return
      }
      setMsg('Great! Level solved ✅', true)
    })

    // Expose reset
    reset()
  }

  // init on DOM ready
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init)
  else init()
})();
