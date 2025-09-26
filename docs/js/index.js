window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/

    bulmaSlider.attach();

})

// 图片放大器：支持点击打开、滚轮缩放、按钮缩放、ESC 关闭
(function(){
  const MIN_SCALE = 1;
  const MAX_SCALE = 4;
  const STEP = 0.12;

  // 插入 modal DOM（只插入一次）
  let modal = document.getElementById('imgModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'imgModal';
    modal.className = 'img-modal';
    modal.innerHTML = `
      <div class="img-modal-backdrop" data-role="backdrop"></div>
      <div class="img-modal-content" role="dialog" aria-modal="true">
        <img src="" alt="" />
        <button class="img-modal-close" aria-label="Close">&times;</button>
        <div class="img-modal-controls">
          <button class="zoom-in" title="放大">＋</button>
          <button class="zoom-out" title="缩小">－</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }

  const backdrop = modal.querySelector('[data-role="backdrop"]');
  const modalImg = modal.querySelector('img');
  const closeBtn = modal.querySelector('.img-modal-close');
  const zoomInBtn = modal.querySelector('.zoom-in');
  const zoomOutBtn = modal.querySelector('.zoom-out');

  let scale = 1;

  function setScale(s){
    scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
    modalImg.style.transform = `scale(${scale})`;
  }

  function openModal(imgEl){
    const src = imgEl.dataset.full || imgEl.src;
    modalImg.src = src;
    modalImg.alt = imgEl.alt || '';
    setScale(1);
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
    // focus for keyboard
    closeBtn.focus();
  }

  function closeModal(){
    modal.classList.remove('open');
    document.body.style.overflow = '';
    // 清掉 src 防止继续占用内存（可选）
    // modalImg.src = '';
  }

  // 绑定点击打开
  document.querySelectorAll('img.zoomable').forEach(img=>{
    img.addEventListener('click', function(e){
      openModal(img);
    });
    // 双击快速放大/还原（桌面）
    img.addEventListener('dblclick', function(){
      openModal(img);
      setTimeout(()=> setScale(2), 50);
    });
  });

  // backdrop 或 close 按钮 关闭
  backdrop.addEventListener('click', closeModal);
  closeBtn.addEventListener('click', closeModal);

  // zoom buttons
  zoomInBtn.addEventListener('click', ()=> setScale(scale + STEP));
  zoomOutBtn.addEventListener('click', ()=> setScale(scale - STEP));

  // wheel 缩放（在 modal 打开时生效）
  modal.addEventListener('wheel', function(e){
    if (!modal.classList.contains('open')) return;
    e.preventDefault();
    const delta = e.deltaY || e.wheelDelta;
    if (delta > 0) setScale(scale - STEP);
    else setScale(scale + STEP);
  }, { passive: false });

  // 键盘：Esc 关闭， +/- 缩放
  window.addEventListener('keydown', function(e){
    if (!modal.classList.contains('open')) return;
    if (e.key === 'Escape') closeModal();
    if (e.key === '+' || e.key === '=') setScale(scale + STEP);
    if (e.key === '-') setScale(scale - STEP);
  });

  // 移动端：双指缩放（可选简单实现）
  let lastTouchDist = null;
  modalImg.addEventListener('touchstart', function(e){
    if (e.touches.length === 2) {
      lastTouchDist = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
    }
  }, {passive:true});

  modalImg.addEventListener('touchmove', function(e){
    if (e.touches.length === 2 && lastTouchDist != null) {
      const currDist = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
      const diff = (currDist - lastTouchDist) / 200; // 调节灵敏度
      setScale(scale + diff);
      lastTouchDist = currDist;
      e.preventDefault();
    }
  }, {passive:false});

  modalImg.addEventListener('touchend', function(e){
    if (e.touches.length < 2) lastTouchDist = null;
  });

})();

