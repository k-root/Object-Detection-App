import {
  Component,
  OnInit,
  AfterViewInit,
  AfterViewChecked,
  AfterContentInit
} from '@angular/core';

// Metadata
export interface RouteInfo {
  path: string;
  title: string;
  type: string;
  collapse?: string;
  icontype: string;
  // icon: string;
  children?: ChildrenItems[];
}

export interface ChildrenItems {
  path: string;
  title: string;
  ab: string;
  type?: string;
}

// Menu Items
export const ROUTES: RouteInfo[] = [
  {
    path: '/show',
    title: 'Train Model',
    type: 'link',
    icontype: 'nc-icon nc-bullet-list-67'
  },
  // {
  //   path: '/add',
  //   title: 'Add Template',
  //   type: 'link',
  //   icontype: 'nc-icon nc-simple-add'
  // },
  {
    path: '/test',
    title: 'Load Model',
    type: 'link',
    icontype: 'nc-icon nc-button-play'
  },
  // {
  //  path: '/configurations',
  //  title: 'configurations',
  //  type: 'link',
  //  icontype: 'fa fa-cogs'
  // },
  // {
  //  path: '/flows',
  //  title: 'Integration',
  // type: 'link',
  //  icontype: 'nc-icon nc-tile-56'
  // },
  // {
  //  path: '/import',
  //  title: 'Import template',
  //  type: 'link',
  //  icontype: 'nc-icon nc-cloud-upload-94'
  // },
  // {
  //     path: '/tables',
  //     title: 'Tables',
  //     type: 'sub',
  //     collapse: 'tables',
  //     icontype: 'nc-icon nc-single-copy-04',
  //     children: [
  //         {path: 'regular', title: 'Regular Tables', ab:'RT'},
  //         {path: 'extended', title: 'Extended Tables', ab:'ET'},
  //         {path: 'datatables.net', title: 'Datatables.net', ab:'DT'}
  //     ]
  // },{
  //     path: '/maps',
  //     title: 'Maps',
  //     type: 'sub',
  //     collapse: 'maps',
  //     icontype: 'nc-icon nc-pin-3',
  //     children: [
  //         {path: 'google', title: 'Google Maps', ab:'GM'},
  //         {path: 'fullscreen', title: 'Full Screen Map', ab:'FSM'},
  //         {path: 'vector', title: 'Vector Map', ab:'VM'}
  //     ]
  // },{
  //     path: '/charts',
  //     title: 'Charts',
  //     type: 'link',
  //     icontype: 'nc-icon nc-chart-bar-32'

  // },{
  //     path: '/calendar',
  //     title: 'Calendar',
  //     type: 'link',
  //     icontype: 'nc-icon nc-calendar-60'
  // }
  // {
  //   path: '/pages',
  //   title: 'Pages',
  //   collapse: 'pages',
  //   type: 'sub',
  //   icontype: 'nc-icon nc-book-bookmark',
  //   children: [
  //     { path: 'timeline', title: 'Timeline Page', ab: 'T' },
  //     { path: 'user', title: 'User Page', ab: 'UP' },
  //     { path: 'login', title: 'Login Page', ab: 'LP' },
  //     { path: 'register', title: 'Register Page', ab: 'RP' },
  //     { path: 'lock', title: 'Lock Screen Page', ab: 'LSP' }
  //   ]
  // }
];

@Component({
  moduleId: module.id,
  selector: 'sidebar-cmp',
  templateUrl: 'sidebar.component.html'
})
export class SidebarComponent {
  public menuItems: any[];
  isNotMobileMenu() {
    if (window.outerWidth > 991) {
      return false;
    }
    return true;
  }

  ngOnInit() {
    this.menuItems = ROUTES.filter(menuItem => menuItem);
  }
  ngAfterViewInit() {}
}
