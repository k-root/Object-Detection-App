import { Routes } from '@angular/router';

import { ShowTemplatesComponent } from './show-templates.component';

export const ShowTemplatesRoutes: Routes = [{
    path: '',
    children: [{
        path: '',
        component: ShowTemplatesComponent
    },
    // {
    //     path: 'selectModel',
    //     component: SelectTrainModelComponent
    // }
]
}];
