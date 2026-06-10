//! #277 parity floor: representative CollectionWrapper ops through the anki crate.
use anki::collection::CollectionBuilder;
use anki_proto::notes::note_fields_check_response::State;

fn main() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join("anki-spike-col");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("collection.anki2");
    let _ = std::fs::remove_file(&path);

    // 1. Open (creates) a collection — the CollectionWrapper.open equivalent.
    let mut col = CollectionBuilder::new(&path).build()?;

    // 2. upsert path: resolve the Basic notetype, create a note, duplicate-check it.
    let basic = col.get_notetype_by_name("Basic")?.expect("Basic exists");
    let mut note = basic.new_note();
    note.set_field(0, "What is a mitochondrion?")?;
    note.set_field(1, "The powerhouse of the cell")?;
    let deck_id = anki::decks::DeckId(1); // Default deck
    col.add_note(&mut note, deck_id)?;
    println!("created note id={}", note.id);

    // fields_check: a second note with the same first field must report duplicate.
    let mut dup = basic.new_note();
    dup.set_field(0, "What is a mitochondrion?")?;
    dup.set_field(1, "different back")?;
    let state = col.note_fields_check(&dup)?;
    println!("duplicate check: {:?}", state);
    assert!(matches!(state, State::Duplicate));

    // 3. read path: find_notes via the search interface + read fields back.
    let nids = col.search_notes_unordered("deck:*")?;
    println!("found {} notes", nids.len());
    assert_eq!(nids.len(), 1);
    let stored = col.storage.get_note(nids[0])?.expect("note exists");
    println!("fields: {:?}", stored.fields());
    assert_eq!(stored.fields()[1], "The powerhouse of the cell");

    // 4. media op: write a file through the media manager.
    let media_dir = dir.join("collection.media");
    std::fs::create_dir_all(&media_dir)?;
    let mgr = anki::media::MediaManager::new(&media_dir, dir.join("media.db"))?;
    let name = mgr.add_file("hello.txt", b"hi")?;
    println!("media stored as: {}", name);

    // 5. col.mod — the drift watermark Shrike leans on. The direct accessor is
    // pub(crate); the public path is the timestamps cache on Collection state
    // (or the protobuf service layer) — recorded as a coverage finding.
    println!("changed since sync: {}", col.timing_today()?.days_elapsed);

    // 6. strip_html — embed_text's pinned dependency.
    let stripped = anki::text::strip_html("a<br>b <b>bold</b>");
    println!("strip_html: {stripped:?}");
    Ok(())
}
