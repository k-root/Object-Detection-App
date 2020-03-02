import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { LearnModeComponent	 } from './learn-mode.component';

declare interface DataTable {
    headerRow: string[];
    footerRow: string[];
    dataRows: string[][];
}

@NgModule({
    imports: [
        CommonModule,
        FormsModule
    ],
    exports: [LearnModeComponent],
    declarations: [LearnModeComponent]
})

export class LearnModeModule {}
