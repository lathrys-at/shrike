//! Tauri v2 shell entry. Minimal evidence-only desktop wrapper: it stands up a
//! tray icon and a window over the SPA's `frontendDist`. In the real desktop
//! app this is where the daemon lifecycle (spawn/own the server build
//! invisibly, surface logs) and the native dialogs live (distribution.md
//! profile 2).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::tray::TrayIconBuilder;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // A tray icon proves the Tauri-native integration axis the spike
            // judges (tray / dialogs / lifecycle). The real app wires a menu
            // (Open, Status, Quit) and the daemon supervisor here.
            let _tray = TrayIconBuilder::new().tooltip("Shrike").build(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the Shrike desktop shell");
}
