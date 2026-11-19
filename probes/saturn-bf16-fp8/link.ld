/* Minimal bare-metal layout for Spike's default RAM at 0x8000_0000. */
SECTIONS
{
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
}
