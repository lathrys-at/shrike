/* The embedded smoke host (#333): a plain C program embeds the Shrike kernel
 * — zero CPython, zero reimplemented logic — and runs open → upsert → search
 * → close against a temp collection. Built by the MANUAL Bazel target
 * //native/shrike-cabi:smoke_host (kept off the per-PR lanes).
 */
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The surface under test (matches native/shrike-cabi/src/lib.rs). */
typedef struct ShrikeKernel ShrikeKernel;
extern ShrikeKernel *shrike_kernel_open(const char *collection_path, const char *cache_dir);
extern char *shrike_upsert_notes_json(ShrikeKernel *k, const char *notes_json,
                                      const char *on_duplicate, bool dry_run);
extern char *shrike_search(ShrikeKernel *k, const char *query, size_t top_k);
extern char *shrike_index_status_json(ShrikeKernel *k);
extern int shrike_kernel_close(ShrikeKernel *k);
extern const char *shrike_last_error(void);
extern void shrike_string_free(char *s);
extern int shrike_set_log_callback(void (*cb)(unsigned char level, const char *target,
                                              const char *msg));

static int log_lines = 0;

static void on_log(unsigned char level, const char *target, const char *msg) {
  (void)level;
  (void)target;
  (void)msg;
  log_lines++;
}

static void fail(const char *what) {
  const char *err = shrike_last_error();
  fprintf(stderr, "FAIL %s: %s\n", what, err ? err : "(no error message)");
  exit(1);
}

int main(void) {
  shrike_set_log_callback(on_log);

  char dir[512];
  const char *tmp = getenv("TEST_TMPDIR");
  if (tmp == NULL) tmp = "/tmp";
  snprintf(dir, sizeof dir, "%s/shrike-c-smoke", tmp);
  char col[600], cache[600];
  snprintf(col, sizeof col, "%s/c.anki2", dir);
  snprintf(cache, sizeof cache, "%s/cache", dir);
  char mkdir_cmd[600];
  snprintf(mkdir_cmd, sizeof mkdir_cmd, "mkdir -p %s", dir);
  if (system(mkdir_cmd) != 0) fail("mkdir");

  ShrikeKernel *k = shrike_kernel_open(col, cache);
  if (k == NULL) fail("open");

  char *results = shrike_upsert_notes_json(
      k,
      "[{\"note_type\": \"Basic\", \"deck\": \"Default\","
      "  \"fields\": {\"Front\": \"the mitochondria powerhouse\", \"Back\": \"atp\"}}]",
      "error", false);
  if (results == NULL) fail("upsert");
  if (strstr(results, "\"created\"") == NULL) {
    fprintf(stderr, "FAIL upsert results: %s\n", results);
    return 1;
  }
  shrike_string_free(results);

  char *hits = shrike_search(k, "mitochondria powerhouse", 5);
  if (hits == NULL) fail("search");
  if (strstr(hits, "note_id") == NULL) {
    fprintf(stderr, "FAIL search hits: %s\n", hits);
    return 1;
  }
  shrike_string_free(hits);

  char *status = shrike_index_status_json(k);
  if (status == NULL) fail("status");
  shrike_string_free(status);

  if (shrike_kernel_close(k) != 0) fail("close");

  if (log_lines == 0) {
    fprintf(stderr, "FAIL: no kernel log lines reached the host sink\n");
    return 1;
  }
  printf("OK: embedded C host drove the kernel end to end (%d log lines)\n", log_lines);
  return 0;
}
