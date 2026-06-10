"""Inline JavaScript snippets injected by the chat shell via st.iframe.

These live here (not in ``chat_shell.py``) so the Python module stays focused
on the Streamlit layout, and so the JS itself is easier to scan without 900
lines of Python noise on either side.
"""
from __future__ import annotations


DRAG_BRIDGE_SCRIPT = """
<script>
(function () {
    var doc = window.parent && window.parent.document
        ? window.parent.document : document;
    var win = window.parent || window;
    var root = doc.documentElement;
    var bridgeVersion = "home-drop-native-v10";
    if (win.__dwDragBridge && win.__dwDragBridge.version === bridgeVersion) {
        return;
    }
    if (win.__dwDragBridge && win.__dwDragBridge.handlers) {
        try {
            var old = win.__dwDragBridge.handlers;
            doc.removeEventListener("dragenter", old.dragenter, true);
            doc.removeEventListener("dragover", old.dragover, true);
            doc.removeEventListener("dragleave", old.dragleave, true);
            doc.removeEventListener("drop", old.drop, true);
            doc.removeEventListener("dragend", old.dragend, true);
            win.removeEventListener("blur", old.blur);
            if (win.__dwDragBridge.resizeObserver) {
                win.__dwDragBridge.resizeObserver.disconnect();
            }
            if (win.__dwDragBridge.mutationObserver) {
                win.__dwDragBridge.mutationObserver.disconnect();
            }
            if (win.__dwDragBridge.sidebarInterval) {
                win.clearInterval(win.__dwDragBridge.sidebarInterval);
            }
        } catch (e) {}
    }
    root.dataset.dwDragBridgeReady = bridgeVersion;
    var depth = 0;
    var hideTimer = null;
    var resizeObserver = null;
    var routingDrop = false;
    var sidebarInterval = null;
    function sidebarOffset() {
        var expanded = doc.querySelector('[data-testid="stSidebar"][aria-expanded="true"]');
        if (expanded) {
            var rect = expanded.getBoundingClientRect();
            if (rect && rect.width > 80) {
                return Math.max(64, Math.round(rect.right || rect.width));
            }
        }
        return 64;
    }
    function syncSidebarOffset() {
        root.style.setProperty("--dw-sidebar-offset", sidebarOffset() + "px");
    }
    function hasFiles(event) {
        var types = event.dataTransfer && event.dataTransfer.types;
        if (!types) {
            return false;
        }
        return Array.prototype.indexOf.call(types, "Files") !== -1;
    }
    function isHomeArea(event) {
        return !event.clientX || event.clientX >= sidebarOffset();
    }
    function show() {
        syncSidebarOffset();
        if (hideTimer) {
            window.clearTimeout(hideTimer);
            hideTimer = null;
        }
        root.setAttribute("data-dw-file-drag", "1");
    }
    function getUploaderHost() {
        return doc.querySelector('[class*="st-key-home_drop_upload"]');
    }
    function getUploaderDropzone() {
        return doc.querySelector(
            '[class*="st-key-home_drop_upload"] [data-testid="stFileUploaderDropzone"]'
        ) || getUploaderHost();
    }
    function getUploaderInput() {
        return doc.querySelector(
            '[class*="st-key-home_drop_upload"] input[type="file"]'
        );
    }
    function cloneTransfer(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length) {
            return null;
        }
        try {
            var transfer = new win.DataTransfer();
            Array.prototype.forEach.call(dataTransfer.files, function (file) {
                transfer.items.add(file);
            });
            return transfer;
        } catch (cloneError) {
            return dataTransfer;
        }
    }
    function filesAlreadyShown(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length || !doc.body) {
            return false;
        }
        var text = doc.body.innerText || "";
        return Array.prototype.every.call(dataTransfer.files, function (file) {
            return file && file.name && text.indexOf(file.name) !== -1;
        });
    }
    function pushFilesToUploader(dataTransfer) {
        if (!dataTransfer || !dataTransfer.files || !dataTransfer.files.length) {
            return false;
        }
        var input = getUploaderInput();
        if (!input) {
            return false;
        }
        var transfer = cloneTransfer(dataTransfer);
        var files = transfer && transfer.files ? transfer.files : dataTransfer.files;
        try { input.value = ""; } catch (clearError) {}
        try {
            var descriptor = Object.getOwnPropertyDescriptor(
                win.HTMLInputElement.prototype,
                "files"
            );
            if (descriptor && descriptor.set) {
                descriptor.set.call(input, files);
            } else {
                input.files = files;
            }
        } catch (error) {
            try {
                Object.defineProperty(input, "files", {
                    value: files,
                    configurable: true
                });
            } catch (innerError) {
                return false;
            }
        }
        input.dispatchEvent(new win.Event("input", { bubbles: true, composed: true }));
        input.dispatchEvent(new win.Event("change", { bubbles: true, composed: true }));
        return true;
    }
    function forwardDropToUploader(dataTransfer) {
        var dropzone = getUploaderDropzone();
        var transfer = cloneTransfer(dataTransfer);
        if (!dropzone || !transfer || !transfer.files || !transfer.files.length) {
            return false;
        }
        routingDrop = true;
        try {
            ["dragenter", "dragover", "drop"].forEach(function (type) {
                var event = new win.DragEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    composed: true,
                    dataTransfer: transfer
                });
                dropzone.dispatchEvent(event);
            });
            return true;
        } catch (error) {
            return false;
        } finally {
            routingDrop = false;
        }
    }
    function hide() {
        root.removeAttribute("data-dw-file-drag");
        depth = 0;
    }
    function hideSoon() {
        if (hideTimer) {
            window.clearTimeout(hideTimer);
        }
        hideTimer = window.setTimeout(hide, 80);
    }
    function onDragEnter(event) {
        if (!hasFiles(event) || !isHomeArea(event)) {
            return;
        }
        depth += 1;
        show();
    }
    // We must call preventDefault on dragover so the browser knows the
    // page is a valid drop target — without it, drop never fires.
    function onDragOver(event) {
        if (!hasFiles(event) || !isHomeArea(event)) {
            return;
        }
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        syncSidebarOffset();
        show();
    }
    function onDragLeave(event) {
        if (!hasFiles(event)) {
            return;
        }
        depth = Math.max(0, depth - 1);
        if (depth <= 0) {
            hideSoon();
        }
    }
    function onDrop(event) {
        if (routingDrop) {
            return;
        }
        if (!hasFiles(event) || !isHomeArea(event)) {
            hide();
            return;
        }
        syncSidebarOffset();
        var transfer = cloneTransfer(event.dataTransfer) || event.dataTransfer;
        var host = getUploaderHost();
        var isNativeTarget = host && host.contains(event.target);
        event.preventDefault();
        try { event.dataTransfer.dropEffect = "copy"; } catch (e) {}

        if (isNativeTarget) {
            // Let Streamlit/react-dropzone receive the original drop event too.
            window.setTimeout(function () {
                if (!filesAlreadyShown(transfer)) {
                    pushFilesToUploader(transfer);
                }
            }, 220);
            window.setTimeout(hide, 520);
            return;
        }

        event.stopPropagation();
        if (!forwardDropToUploader(transfer)) {
            pushFilesToUploader(transfer);
        }
        window.setTimeout(function () {
            if (!filesAlreadyShown(transfer)) {
                pushFilesToUploader(transfer);
            }
            hide();
        }, 220);
    }
    syncSidebarOffset();
    sidebarInterval = win.setInterval(syncSidebarOffset, 250);
    try {
        resizeObserver = new ResizeObserver(syncSidebarOffset);
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            resizeObserver.observe(sidebar);
        }
        resizeObserver.observe(doc.body);
    } catch (e) {}
    var mutationObserver = null;
    try {
        mutationObserver = new MutationObserver(syncSidebarOffset);
        mutationObserver.observe(doc.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["aria-expanded", "style", "class"]
        });
    } catch (e) {}
    doc.addEventListener("dragenter", onDragEnter, true);
    doc.addEventListener("dragover", onDragOver, true);
    doc.addEventListener("dragleave", onDragLeave, true);
    doc.addEventListener("drop", onDrop, true);
    doc.addEventListener("dragend", hide, true);
    try {
        win.addEventListener("blur", hide);
    } catch (e) {}
    win.__dwDragBridge = {
        version: bridgeVersion,
        handlers: {
            dragenter: onDragEnter,
            dragover: onDragOver,
            dragleave: onDragLeave,
            drop: onDrop,
            dragend: hide,
            blur: hide
        },
        resizeObserver: resizeObserver,
        mutationObserver: mutationObserver,
        sidebarInterval: sidebarInterval
    };
})();
</script>
"""
