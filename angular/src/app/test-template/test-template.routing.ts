import { Routes } from '@angular/router';

import { TestTemplateComponent } from './test-template.component';

export const TestTemplateRoutes: Routes = [{
    path: '',
    children: [{
        path: '',
        component: TestTemplateComponent
    }]
}];
