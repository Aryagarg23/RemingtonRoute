document.addEventListener('DOMContentLoaded',function(){
  const modal = document.getElementById('modal')
  const ctas = [document.getElementById('cta-top'), document.getElementById('cta-hero')]
  const close = document.getElementById('modal-close')
  const modalCta = document.getElementById('modal-cta')
  ctas.forEach(el=>{ if(el) el.addEventListener('click',openModal) })
  if(close) close.addEventListener('click',closeModal)
  if(modalCta) modalCta.addEventListener('click',()=>{ window.location.href='README.md' })
  document.getElementById('year').textContent = new Date().getFullYear()
  function openModal(e){ e && e.preventDefault(); modal.classList.remove('hidden') }
  function closeModal(e){ e && e.preventDefault(); modal.classList.add('hidden') }
})
