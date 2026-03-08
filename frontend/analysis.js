const analysisTabs = Array.from(document.querySelectorAll(".analysis-tab"));
const analysisSections = Array.from(document.querySelectorAll(".analysis-section"));
const defaultTarget = analysisTabs.find((tab) => tab.dataset.target === "gpt")?.dataset.target
  || analysisTabs[0]?.dataset.target;

function setActiveTab(target, options = {}) {
  const name = target || defaultTarget;
  if (!name) return;
  const updateHash = options.updateHash !== false;

  analysisTabs.forEach((tab) => {
    const isActive = tab.dataset.target === name;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  analysisSections.forEach((section) => {
    const isActive = section.dataset.section === name;
    section.classList.toggle("active", isActive);
    section.setAttribute("aria-hidden", String(!isActive));
  });

  if (updateHash) {
    history.replaceState(null, "", `#${name}`);
  }
}

function targetFromHash() {
  const hash = window.location.hash.replace("#", "");
  return analysisTabs.some((tab) => tab.dataset.target === hash) ? hash : null;
}

analysisTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveTab(tab.dataset.target);
  });
});

window.addEventListener("hashchange", () => {
  setActiveTab(targetFromHash(), { updateHash: false });
});

setActiveTab(targetFromHash(), { updateHash: false });
